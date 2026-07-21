# Releasing Rangefinder

PyPI publication uses GitHub Actions trusted publishing, so no long-lived PyPI
token is stored in GitHub.

## One-time PyPI setup

Before publishing the first release, add a pending GitHub publisher at
<https://pypi.org/manage/account/publishing/> with these exact values:

- PyPI project name: `rangefinder`
- GitHub owner: `kylemcdonald`
- Repository: `rangefinder`
- Workflow: `release.yml`
- Environment: `pypi`

## Release checklist

1. Update `__version__` in `src/rangefinder/__init__.py` and commit the change.
2. Run `python -m pytest`.
3. Run `python -m build` and `python -m twine check dist/*`.
4. Tag the exact release commit as `v<version>` and push the tag.
5. Create a GitHub release for that tag. Publishing the release triggers the
   trusted-publishing workflow in `.github/workflows/release.yml`.

PyPI does not allow an uploaded filename/version to be replaced. If a release
fails after files reach PyPI, increment the version before retrying.
