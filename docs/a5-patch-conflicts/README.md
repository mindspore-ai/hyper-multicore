# A5 patch import: unresolved integration work

This draft imports the 27-patch `megamoe_a5_patch.tar.gz` series onto
`3c60e08dbe62073825b5a25568d75e47146a8879`, the first parent of upstream master
`797e8bdbb9fb19ba6e1753236bfe47a1178b1b58`. It has not been rebased onto the newer
master. The user requested a PR retaining unresolved conflicts for follow-up.

The source tree is an incomplete integration snapshot, not a validated build.
No build, unit test or device qualification was run for this import. Qualification
claims in the imported source documents describe historical source results and do
not validate this branch.

## Conflict records

`manifest.json` records the patches requiring intervention. For three-way conflicts,
the incoming patch version was retained; the base, previous branch version and
incoming version are archived as `.txt` files where available. Patch 4's original
diff is retained in `0004.patch`; its previous runtime files are available from
the branch's base commit.

Patches 15, 16, 18 and 24 could only be partially applied. Their numbered directories
retain the original patch, application log and rejected hunks. Errors for missing
prerequisite files may appear only in the application log. These changes still
require integration into the source tree; their presence in the archive does not
mean that the source implements them.

Patch 17 was already represented after the preceding conflict selection and did
not create a separate commit. The remaining patch subjects are retained as commits.

Review the rejected hunks and the overwritten upstream alternatives before marking
the PR ready. GitHub's mergeability result concerns merging this branch into master;
it does not report the unresolved patch-import work recorded here.
