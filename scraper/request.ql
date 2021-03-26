query ($login: String!) {
  user(login: $login) {
    id
    bio
  }
}

{
  "login": "InBinaryWorld"
}

--------------------------------------------------------------------------

query ($login: String!, $userId: ID!, $limit: Int!, $repoCursor: String) {
  user(login: $login) {
    repositories(first: $limit, after: $repoCursor) {
      edges {
        cursor
        node {
          name
          isFork
          description
          dependencyGraphManifests {
            nodes {
              dependencies {
                nodes {
                  packageName
                }
              }
            }
          }
          languages(first: 100, orderBy: {field: SIZE, direction: DESC}) {
            nodes {
              name
            }
          }
          topics: repositoryTopics(first: 100) {
            nodes {
              topic {
                name
              }
            }
          }
          defaultBranchRef {
            target {
              ... on Commit {
                totalCommits: history {
                  totalCount
                }
                userCommits: history(author: {id: $userId}) {
                  totalCount
                }
              }
            }
          }
        }
      }
    }
  }
}



{
  "login": "InBinaryWorld",
  "userId": "MDQ6VXNlcjQzOTU2MDQ0",
  "repoCursor": null
}
