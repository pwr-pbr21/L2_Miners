# Preparing data 

## Selecting data from StackOverflow
https://data.stackexchange.com/stackoverflow/
```
select *
from Users
where WebsiteUrl like '%github%'
```
The query returned 30990 lines.

## Processing data for collecting GitHub usernames
- Removing lines with empty AboutMe section  
    18514 lines remained.
- Applying regexes to retrieve GitHub usernames.
    ```
    https?:\/\/([^\/]+)\.github\.io.*
    https?:\/\/github.com\/([^\/]+)
    https?:\/\/www.github.com\/([^\/]+)
    ```
    17970 lines remained.
