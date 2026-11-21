"""Instagram post monitor built on instaloader.

Checks each account in config.json every `interval_minutes` and downloads any
post newer than the last one seen. State lives in state.json, logs in monitor.log.

Usage:
    python monitor.py          # run forever
    python monitor.py --once   # single check, then exit
"""
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import instaloader

import pipeline
from instaloader.exceptions import (
    ConnectionException,
    LoginRequiredException,
    ProfileNotExistsException,
    QueryReturnedNotFoundException,
    TooManyRequestsException,
)

BASE = Path(__file__).resolve().parent
CONFIG_FILE = BASE / "config.json"
STATE_FILE = BASE / "state.json"
LOG_FILE = BASE / "monitor.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("monitor")


def load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as e:
        log.warning("%s is not valid JSON (%s); using previous settings", path.name, e)
        return default


def save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def make_loader(cfg):
    download_dir = (BASE / cfg.get("download_dir", "downloads")).as_posix()
    L = instaloader.Instaloader(
        dirname_pattern=download_dir + "/{target}",
        download_videos=cfg.get("download_videos", True),
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        post_metadata_txt_pattern="{caption}" if cfg.get("save_captions", True) else "",
        quiet=True,
    )
    ensure_login(L, cfg)
    return L


def ask(prompt):
    if not sys.stdin or not sys.stdin.isatty():
        return ""
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


BROWSERS = ["firefox", "chrome", "edge", "brave", "opera", "vivaldi", "chromium"]


def has_console():
    return bool(sys.stdin) and sys.stdin.isatty()


def login_from_browser(L, browser):
    """Copy the Instagram login cookies from a local browser into instaloader.

    Runs only on this machine; the cookies go straight into the instaloader session.
    Returns the logged-in username, or None on failure.
    """
    try:
        import browser_cookie3
    except ImportError:
        log.error("Browser login needs an extra package: python -m pip install browser_cookie3")
        return None
    try:
        jar = getattr(browser_cookie3, browser)(domain_name="instagram.com")
    except Exception as e:  # locked DB, unsupported encryption, browser missing...
        log.error("Could not read the Instagram login from %s: %s", browser, e)
        if browser in ("chrome", "edge", "brave"):
            log.error("Newer Chrome-based browsers lock their cookies. Try Firefox, or close %s and retry.", browser)
        return None
    cookies = {c.name: c.value for c in jar if "instagram" in c.domain}
    if "sessionid" not in cookies:
        log.error("You're not logged in to instagram.com in %s. Log in there first, then retry.", browser)
        return None
    if "csrftoken" not in cookies:
        log.error("The %s login is incomplete (no csrftoken). Reload instagram.com in %s, close it, then retry.",
                  browser, browser)
        return None
    # load_session builds a properly-headered logged-in session (incl. X-CSRFToken);
    # update_cookies alone leaves the anonymous headers, which Instagram rejects.
    L.context.load_session(None, cookies)
    username = whoami(L)
    if not username:
        log.error("Instagram rejected the %s login. Open instagram.com in %s, make sure it works, then retry.",
                  browser, browser)
        return None
    L.context.username = username
    return username


def whoami(L):
    """Username of the logged-in session, or None. Tries the mobile API first (it works
    with browser sessions); falls back to instaloader's legacy GraphQL check."""
    uid = L.context._session.cookies.get("ds_user_id")
    if uid:
        try:
            data = L.context.get_iphone_json(f"api/v1/users/{uid}/info/", {})
            return data["user"]["username"]
        except Exception as e:
            log.debug("Mobile API username lookup failed: %s", e)
    try:
        return L.test_login()
    except Exception:
        return None


def login_with_password(L, user):
    quiet = L.context.quiet
    L.context.quiet = False  # interactive_login refuses to prompt in quiet mode
    try:
        for attempt in range(2):
            try:
                L.interactive_login(user)  # prompts for password and 2FA code if needed
                return user
            except instaloader.exceptions.LoginException as e:
                msg = str(e)
                if "Checkpoint required" not in msg or attempt == 1:
                    log.error("Login failed: %s", msg)
                    return None
                path = msg.split("Point your browser to", 1)[1].split(" - follow the", 1)[0].strip()
                print("\nInstagram wants to verify this login. Open this link in a browser where you're")
                print("logged in as " + user + ", follow the steps (usually approving \"This was me\"):\n")
                print("    https://www.instagram.com" + path + "\n")
                ask("When you've finished, press Enter to try again...")
    except EOFError:
        log.error("Login needs console input. Start via start-monitor.bat.")
        return None
    finally:
        L.context.quiet = quiet
    return None


def ensure_login(L, cfg):
    """Reuse a saved session if valid; otherwise log in from the browser or with a password."""
    user = cfg.get("login_user")
    if user:
        try:
            L.load_session_from_file(user)
            if whoami(L) == user:
                log.info("Logged in as %s (saved session)", user)
                return
            log.warning("Saved session for %s has expired", user)
        except FileNotFoundError:
            pass

    browser = cfg.get("login_browser")
    # A browser login can refresh itself with no one at the keyboard.
    if browser:
        got = login_from_browser(L, browser)
        if got:
            return finish_login(L, cfg, got, browser)
        if not has_console():
            sys.exit(1)

    if not has_console():
        log.error("Login needed but no console to type in. Start via start-monitor.bat.")
        sys.exit(1)

    print("\nHow do you want to log in to Instagram?")
    print("  1) Use the login from my web browser (recommended, avoids security checkpoints)")
    print("  2) Type my username and password")
    choice = ask("Choose 1 or 2: ") or "1"
    if choice == "1":
        print("Log in to instagram.com in your browser first if you haven't.")
        b = (ask("Which browser? (" + ", ".join(BROWSERS) + ") [firefox]: ") or "firefox").lower()
        if b not in BROWSERS:
            log.error("Unknown browser %r", b)
            sys.exit(1)
        got = login_from_browser(L, b)
        if got:
            return finish_login(L, cfg, got, b)
        sys.exit(1)

    user = user or ask("Instagram username: ").lstrip("@")
    if not user:
        log.warning("Running logged out; Instagram will likely block most requests")
        return
    got = login_with_password(L, user)
    if not got:
        print("Tip: choose option 1 next time to use your browser login instead.")
        sys.exit(1)
    finish_login(L, cfg, got, None)


def finish_login(L, cfg, user, browser):
    L.save_session_to_file()
    changed = cfg.get("login_user") != user or cfg.get("login_browser") != browser
    cfg["login_user"], cfg["login_browser"] = user, browser
    if changed:
        save_config(cfg)
    log.info("Logged in as %s; session saved, so you won't be asked again", user)


def ensure_accounts(cfg):
    if cfg.get("accounts"):
        return
    raw = ask("Accounts to monitor (comma-separated): ")
    cfg["accounts"] = [a.strip().lstrip("@") for a in raw.split(",") if a.strip()]
    if not cfg["accounts"]:
        log.error("No accounts to monitor. Add them to %s", CONFIG_FILE)
        sys.exit(1)
    save_config(cfg)


def check_account(L, username, state, cfg):
    """Download posts newer than the stored timestamp. Returns number of new posts."""
    profile = instaloader.Profile.from_username(L.context, username)
    # from_username already has the user id and privacy flag from the profile page.
    # Skip instaloader's extra profile-details query (doc_id 27937681195819736 on
    # graphql/query): Instagram now answers it with 400 "feedback_required" even for
    # healthy accounts. get_posts() itself uses the newer api/graphql endpoint.
    if profile._node.get("is_private"):
        log.info("[%s] profile is private; skipping", username)
        return 0
    profile._has_full_metadata = True
    last_ts = state.get(username)  # ISO timestamp of newest post seen
    last_dt = datetime.fromisoformat(last_ts) if last_ts else None

    new_posts = []
    for post in profile.get_posts():
        post_dt = post.date_utc.replace(tzinfo=timezone.utc)
        if last_dt is None:
            # First time seeing this account.
            if not cfg.get("download_existing_on_first_run", False):
                if post.is_pinned:
                    continue  # pinned posts can be old; find the real newest
                state[username] = post_dt.isoformat()
                log.info("[%s] baseline set at %s (existing posts skipped)", username, post_dt)
                return 0
            new_posts.append(post)
            continue
        if post_dt <= last_dt:
            if post.is_pinned:
                continue  # pinned posts sit on top out of order
            break
        new_posts.append(post)

    if last_dt is None and not new_posts:
        state[username] = datetime.now(timezone.utc).isoformat()
        log.info("[%s] no posts yet; watching from now", username)
        return 0

    # Oldest first so the timestamp advances safely even if we get interrupted.
    for post in sorted(new_posts, key=lambda p: p.date_utc):
        log.info("[%s] NEW POST %s  https://instagram.com/p/%s/", username, post.date_utc, post.shortcode)
        L.download_post(post, target=username)
        stem = post.date_utc.strftime("%Y-%m-%d_%H-%M-%S") + "_UTC"
        post_dir = BASE / cfg.get("download_dir", "downloads") / username
        pipeline.process_post(sorted(post_dir.glob(stem + "*")), username, cfg)
        state[username] = post.date_utc.replace(tzinfo=timezone.utc).isoformat()
        save_state(state)
    return len(new_posts)


BLOCK_MARKERS = ("feedback_required", "checkpoint", "challenge_required", "login_required")


def run_cycle(L, cfg, state):
    """Check every account once. Returns True if Instagram blocked us and we should back off."""
    total = 0
    for username in cfg["accounts"]:
        username = username.strip().lstrip("@")
        if not username:
            continue
        try:
            total += check_account(L, username, state, cfg)
            save_state(state)
        except ProfileNotExistsException:
            log.warning("[%s] profile does not exist", username)
        except LoginRequiredException:
            log.warning("[%s] session no longer valid; logging in again", username)
            ensure_login(L, cfg)
        except QueryReturnedNotFoundException as e:
            log.warning("[%s] needs login or is private: %s", username, e)
        except TooManyRequestsException:
            log.warning("Rate limited by Instagram (429); stopping this cycle")
            return True
        except instaloader.exceptions.AbortDownloadException as e:
            # Not an InstaloaderException subclass, so it must be caught on its own.
            if any(m in str(e) for m in BLOCK_MARKERS):
                log.warning("Instagram has temporarily restricted this account: %s", e)
                return True
            log.warning("[%s] aborted: %s", username, e)
        except ConnectionException as e:
            log.warning("[%s] connection error: %s", username, e)
        except instaloader.exceptions.InstaloaderException as e:
            hint = "" if cfg.get("login_user") else " (Instagram often blocks logged-out access; set login_user)"
            log.warning("[%s] %s: %s%s", username, type(e).__name__, e, hint)
        except Exception as e:  # never let one odd response kill a long-running monitor
            log.exception("[%s] unexpected error: %s", username, e)
        time.sleep(random.uniform(5, 15))  # be gentle between accounts
    log.info("Cycle done: %d new post(s)", total)
    return False


def main():
    once = "--once" in sys.argv
    cfg = load_json(CONFIG_FILE, {}) or {}
    ensure_accounts(cfg)
    state = load_json(STATE_FILE, {})
    L = make_loader(cfg)
    log.info("Monitoring %d account(s) every %s min", len(cfg["accounts"]), cfg.get("interval_minutes", 30))

    blocks = 0
    while True:
        blocked = run_cycle(L, cfg, state)
        if once:
            break
        if blocked:
            # Back off hard; hammering a restricted account extends the block.
            blocks += 1
            hours = min(cfg.get("block_cooldown_hours", 6) * blocks, 24)
            log.warning("Pausing all checks for %d hour(s). Using Instagram normally in your browser "
                        "meanwhile can help lift the restriction.", hours)
            time.sleep(hours * 3600)
            cfg = load_json(CONFIG_FILE, cfg)
            continue
        blocks = 0
        base = cfg.get("interval_minutes", 30) * 60
        wait = base + random.uniform(-0.1, 0.1) * base  # jitter so requests don't look robotic
        log.info("Next check in %.1f min", wait / 60)
        time.sleep(wait)
        # Pick up config edits (new accounts, interval) without restarting.
        cfg = load_json(CONFIG_FILE, cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Stopped")
