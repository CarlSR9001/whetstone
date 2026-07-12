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
- The app writes no request bodies and Nginx logs only the usual method, path,
  status, and connection metadata.
- HTTP exists only for ACME renewal and redirects all other traffic to HTTPS.

## Release layout

```text
/opt/whetstone-tools/releases/<commit>/
/opt/whetstone-tools/current -> releases/<commit>
/etc/systemd/system/whetstone-tools.service
/etc/nginx/sites-available/whetstone-tools
/etc/nginx/sites-enabled/whetstone-tools
/etc/nginx/conf.d/whetstone-rate-limit.conf
```

Switching `current` is atomic. A release never overwrites another release or the
personal site.

## Preflight and release

1. Record the SHA-256 of the existing apex vhost and homepage body.
2. Back up the Nginx and systemd configuration on the VPS with mode `0600`.
3. Run the complete local test suite, JavaScript syntax check, and browser smoke.
4. Commit locally and create the upload with `git archive`; do not package the
   working tree.
5. Extract into a new commit-named release, point `current` at it, install the
   unit, and verify `http://127.0.0.1:8988/api/health` on the VPS.
6. Install the bootstrap vhost, validate Nginx, issue a host-specific certificate
   with the ACME webroot, install the final vhost, validate again, then reload.
7. Verify HTTPS, all eight API samples, certificate hostname, security headers,
   service sandbox score, and unchanged apex checksums.

Every Nginx change is staged into a temporary file first and accepted only after
`nginx -t`. Reloads preserve existing connections.

## Rollback

```bash
sudo rm -f /etc/nginx/sites-enabled/whetstone-tools
sudo rm -f /etc/nginx/conf.d/whetstone-rate-limit.conf
sudo nginx -t && sudo systemctl reload nginx
sudo systemctl disable --now whetstone-tools.service
```

For an application-only rollback, leave Nginx and the unit in place, repoint the
symlink to the prior release, and restart the isolated service:

```bash
sudo ln -sfn /opt/whetstone-tools/releases/<prior-commit> /opt/whetstone-tools/current
sudo systemctl restart whetstone-tools.service
```

The dedicated certificate can remain for renewal or be removed separately after
the vhost is gone. Neither rollback path edits the apex `cyberelf.link` vhost.
