# Whetstone Tools deployment

The public toolbox is one deliberately narrow service at
`whetstone.cyberelf.link`. It does not share a process, port, certificate, vhost,
or document root with the personal site.

## Boundaries

- Nginx is the only public listener. The Python process binds to
  `127.0.0.1:8988`.
- The systemd unit runs as an unprivileged user with a read-only filesystem,
  empty Linux capabilities, resource limits, and no access to home directories.
- The archive is built from a reviewed Git commit. Gitignored `.bcv_runs`, tools,
  private results, browser output, and local secrets cannot enter it.
- The workbench and disposable report card persist no request bodies or
  results. Open Promotion Bench writes only an explicitly published sanitized
  receipt and self-attested manifests, never task contents or answer patches.
  The app also retains operational counters and salted client hashes; Nginx
  retains standard access metadata such as address, request path, status,
  referrer, and user agent.
- HTTP exists only for ACME renewal and redirects all other traffic to HTTPS.

## Release layout

```text
/opt/whetstone-tools/releases/<commit>/
/opt/whetstone-tools/current -> releases/<commit>
/etc/systemd/system/whetstone-tools.service
/etc/systemd/system/whetstone-forge.service
/etc/whetstone-tools/release.env
/etc/nginx/sites-available/whetstone-tools
/etc/nginx/sites-enabled/whetstone-tools
/etc/nginx/conf.d/whetstone-rate-limit.conf
/etc/letsencrypt/renewal-hooks/deploy/whetstone-reload-nginx
```

Switching `current` is atomic. Tools and forge import from that same immutable
release; their persistent state remains outside it. A release never overwrites
another release or the personal site.

## Preflight and release

From a clean, reviewed `main` checkout:

```powershell
powershell.exe -NoProfile -File scripts/release_check.ps1
```

That gate runs the full suite, publication audit, Python/JavaScript/Bash syntax
checks, wheel build, and a dependency-free clean-install smoke that serves the
packaged UI and evidence.

Publish from WSL or another Bash environment:

```bash
bash deploy/publish.sh vps2 HEAD
```

For the one-time upgrade from the pre-lock forge, first verify no refinery child
is running, stop the old forge, and preserve its intended rollback state:

```bash
ssh vps2 sudo systemctl stop whetstone-forge.service
WHETSTONE_EXPECT_FORGE_ACTIVE=active bash deploy/publish.sh vps2 HEAD
```

All later releases coordinate directly through `forge_cycle.lock`.

`publish.sh` refuses a dirty or non-`main` checkout, builds only with
`git archive`, proves that Git expanded the embedded full commit, uploads the
archive, and invokes `install-release.sh` with `sudo`.

The server-side installer:

1. Refuses an active forge refinery child instead of interrupting a mining
   cycle and serializes activation with release and forge-cycle locks.
2. Verifies the embedded commit before accepting a release directory.
3. Records the old symlink, units, release environment, and both state trees.
4. Stops forge and tools, installs both units, and atomically swaps `current`.
5. Starts forge and requires an atomic post-sync status receipt with the same
   version, full commit, systemd PID, and a stable process.
6. Starts tools only after that library sync, then requires matching version,
   full commit, eight core tools, the stateless workbench/private-bank boundary,
   a successful report-card warm-up, and a configured Open Promotion Bench
   publication ledger that never retains tasks or answers.
7. Automatically restores the previous symlink and units if any activation
   gate fails.

After the local service gates pass, the installer atomically updates only the
dedicated Whetstone vhost, requires `nginx -t`, and reloads Nginx. Verify public
HTTPS, MCP initialize, all eight REST samples, certificate hostname, security
headers, the service sandbox score, and the unchanged `cyberelf.link` apex.

Every Nginx change is staged into a temporary file first and accepted only after
`nginx -t`. Reloads preserve existing connections.

## Rollback

```bash
sudo rm -f /etc/nginx/sites-enabled/whetstone-tools
sudo rm -f /etc/nginx/conf.d/whetstone-rate-limit.conf
sudo rm -f /etc/letsencrypt/renewal-hooks/deploy/whetstone-reload-nginx
sudo nginx -t && sudo systemctl reload nginx
sudo systemctl disable --now whetstone-tools.service
```

For a deliberate application rollback, publish the prior full commit through
the same reviewed archive path:

```bash
bash deploy/publish.sh vps2 <prior-full-commit>
```

The installer also performs code, unit, environment, and Nginx rollback
automatically if activation fails. State snapshots are retained for manual
disaster recovery but are not automatically restored; this release does not
change a persisted state schema. Release and state backups live under
`/opt/whetstone-tools/releases` and `/var/backups/whetstone-tools`; neither
path edits the apex `cyberelf.link` vhost.
