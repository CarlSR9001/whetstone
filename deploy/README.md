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

## Receipt signing key

The installer creates one persistent Ed25519 key at
`/var/lib/whetstone-tools/receipt_signing_key` when none exists; later releases
reuse it and add its public key to
`/var/lib/whetstone-tools/receipt_trusted_keys`. The private key is mode 0600,
owned by the unprivileged service user, and never enters a source archive or
release environment file. Public and retired keys are served at
`/.well-known/whetstone-receipt-keys.json`.

For rotation, stop the toolbox, generate a replacement at the same path as the
service user, retain the old public key in `receipt_trusted_keys`, then restart
and verify both a newly signed receipt and an archival receipt before removing
any old private-key backup. Default verification accepts only the active key;
use `verify-receipt --allow-retired --allow-expired` deliberately for an
archival receipt. Revoked keys are never accepted. A signature authenticates
the Whetstone result and build identity; caller-supplied model and harness
labels remain self-attested.

## Preflight and release

From a clean, reviewed `main` checkout:

```powershell
powershell.exe -NoProfile -File scripts/release_check.ps1
```

That gate runs the full suite, publication audit, Python/JavaScript/Bash syntax
checks, wheel build, and a dependency-free clean-install smoke that serves the
packaged UI and evidence.

Publish directly from Windows PowerShell. Archive and clean-tree checks use the
native Windows Git checkout, while the default WSL transport reuses the
operator's `vps2` SSH alias from Ubuntu:

```powershell
powershell.exe -NoProfile -File deploy/publish.ps1 -SshHost vps2 -GitRef HEAD -SshTransport Wsl
```

Use `-SshTransport Native` only when the same host alias and credentials are
configured in Windows OpenSSH.

Or publish from a clean Linux/WSL checkout:

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

Both publishers refuse a dirty or non-`main` checkout, build only with
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
   publication ledger that never retains tasks or answers. It also requires the
   persistent Ed25519 receipt signer to be configured and ready.
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
