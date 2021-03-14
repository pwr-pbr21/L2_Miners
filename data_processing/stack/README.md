# Preparing data 

## Selecting data from StackOverflow
https://data.stackexchange.com/stackoverflow/
```
select *
from Users
where WebsiteUrl like '%github%'
```
The query returned 30990 rows.

## Processing data for collecting GitHub usernames
- Removing rows with empty AboutMe section  
    18514 rows remained.
- Applying regexes to retrieve GitHub usernames.
    ```
    https?:\/\/([^\/]+)\.github\.io.*
    https?:\/\/github.com\/([^\/]+)
    https?:\/\/www.github.com\/([^\/]+)
    ```
    17970 rows remained.

## Processing data for determining technical roles
- Extracting list of roles via regular expressions
    ```
    {
        'name': 'Frontend',
        'pattern': '.*front.{0,1}end.*'
    },
    {
        'name': 'Backend',
        'pattern': '.*back.{0,1}end.*'
    },
    {
        'name': 'DevOps',
        'pattern': '.*dev.{0,1}ops.*'
    },
    {
        'name': 'DataScience',
        'pattern': '.*data.{0,1}scientist.*'
    },
    {
        'name': 'Mobile',
        'pattern': '.*mobile.*'
    }
    ```
- Removing rows with no roles  
    1841 rows remained.
