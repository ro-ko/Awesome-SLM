# Contribution Guidelines

To add, remove, or change things on this repository please submit a pull request that adheres to the following guidelines:

- To add a paper in **Milestone Papers**, please mention why it is important in LLM literature and sort the papers in chronological order. 
- Models in **LLM Leaderboard** should be sorted by their model size.
- Please follow the existing format to add new terms.

Thank you for your suggestions!

## Updating your PR

A lot of times, making a PR adhere to the standards above can be difficult.
If the maintainers notice anything that we'd like changed, we'll ask you to
edit your PR before we merge it. There's no need to open a new PR, just edit
the existing one. If you're not sure how to do that,
[here is a guide](https://github.com/RichardLitt/knowledge/blob/master/github/amending-a-commit-guide.md)
on the different ways you can update your PR so that we can merge it.

## how to add dynamic citation badge

1. get paper id from semantic scholar paper page 
2. create dynamic badge at [this site](https://shields.io/badges/dynamic-json-badge) with this link: https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=citationCount
3. ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F432bef8e34014d726c674bc458008ac895297b51%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)
