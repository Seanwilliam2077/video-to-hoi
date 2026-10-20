# Publishing immutable Human review deliveries

`tools/publish_human_delivery.py` consumes the actual `upload-manifest.json`
written by `tools/package_human_delivery.py`. Its default operation validates
local bytes, MIME types, the generated asset allowlist, archive checksums and
the source commit, then prints an offline plan. It does not authenticate or
contact GitHub until `--publish` is supplied.

```powershell
python tools/publish_human_delivery.py --manifest <package>/upload-manifest.json --source-commit <full-40-character-commit>
```

The fixed destination is `Seanwilliam2077/video-to-hoi`. The release tag equals
the unique packaged delivery ID (for example, `sam-mediapipe-20261021`); it never
uses or rewrites `main`. IDs use the packager's lowercase letters, digits,
periods, underscores and hyphens, at most 64 characters, with an alphanumeric
first character and no trailing period.
Before creation, the publisher checks that the source commit is available in
the target repository and resolves any existing lightweight or annotated tag
to that exact commit. Push the actual source commit through the authorized Git
workflow first. A published prerelease is created for an in-progress delivery.
Reusing that prerelease later can append newly named episode assets while
preserving all existing asset bytes and release notes.

This append workflow requires the GitHub release field `immutable: false`.
That is distinct from this tool's policy of never modifying existing assets.
If repository-level release immutability is enabled, publication locks further
asset additions; the publisher fails clearly before any asset upload. Such a
repository needs a separate authorized draft workflow that attaches the entire
delivery before publication. The tool never changes repository settings.

```powershell
python tools/publish_human_delivery.py --manifest <package>/upload-manifest.json --source-commit <full-40-character-commit> --receipt-dir <publication-receipts> --publish
```

Authentication uses a process-local `GH_TOKEN` when supplied, otherwise the
existing Git credential helper through captured `git credential fill` output.
Credentials are never printed or saved. Authenticated API requests refuse
redirects, and fresh public download verification uses no authentication.
The tool does not install credentials, request model access, publish dashboard
records, switch branches, create commits, or merge changes.

Every asset is rehashed immediately before upload. Uploads stream the file with
an explicit content type and content length. A lost create/upload response
triggers a server lookup before another identical request. Existing asset names
are reused only when their byte count and server SHA256 digest match, or, when
the digest is absent, a fresh anonymous download proves the exact bytes. Every
successful asset is also fully downloaded anonymously and checked. Generated
`<asset-name>.sha256` sidecars are uploaded and verified in the same way.
No existing asset is deleted or replaced. A conflicting or incomplete remote
asset requires a fresh delivery ID rather than an automatic overwrite.

`publication.json` contains `passed`, the exact `release_base_url`, and an
`assets` list with each asset's `name`, `bytes`, `sha256`, `url` and `verified`
fields. `passed: true` appears only after every manifest asset and its sidecar
passes the remote and anonymous checks. Per-asset states are flushed atomically
under `assets/`; immutable `attempt-*.json` files retain each invocation, and
`release.json` records the exact tag/commit/release binding. Resume with the
same receipt directory: prior local receipts do not bypass re-verification.
Keep receipt directories separate from the immutable packaged assets.

On a failed attempt, already uploaded assets remain intact. Review the durable
receipt and rerun with the same manifest and source commit. A changed asset must
receive a new delivery ID; a new episode can use new names in the existing
delivery. A new invocation's `publication.json` covers that invocation's
manifest, not an assertion that all 30 episodes have finished. Preserve the
earlier immutable attempt receipts when aggregating publication evidence.

Existing release notes remain unchanged unless `--update-body --body-file
<notes-file>` explicitly authorizes an update. `--body-file` alone supplies
notes only when creating a new release. Publication is transport evidence,
not acceptance of reconstruction quality, metric scale, contact or official
submission compliance.

The implementation follows GitHub's official
[release API](https://docs.github.com/en/rest/releases/releases) and
[release asset API](https://docs.github.com/en/rest/releases/assets), including
the separate upload host, asset digest, browser download URL and paginated
asset listing. Mocked tests use only independent synthetic transport fixtures.
