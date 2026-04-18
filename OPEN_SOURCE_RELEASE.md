# Open Source Release Checklist

This repository is technically ready for public publication, but publication is
more than pushing code to a public remote. Use this checklist before opening it
up.

## Repository hygiene

- confirm the final repository name and description
- confirm that the chosen license matches your intended reuse policy
- remove any credentials, private paths, or internal-only notes
- verify that generated artifacts included in the repository are intentional

## Hosting configuration

- enable branch protection on the default branch
- enable GitHub Actions for the repository
- enable private vulnerability reporting
- enable issue templates and pull request templates
- choose whether discussions should be enabled for support questions

## Metadata to revisit before public release

The crate metadata now includes a license, readme, description, keywords, and
categories. Before publishing to a package registry or linking from a public
profile, consider adding:

- `repository`
- `homepage`
- `documentation`

These fields were left unset because no canonical public URL is configured in
this workspace.

## Suggested first release tasks

1. publish the repository
2. enable CI on the default branch
3. validate issue templates and security reporting
4. decide whether benchmark artifacts belong in version control or in releases
5. tag the initial public version
