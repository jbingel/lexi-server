# Backend for Lexi software

## Changelog


### Version 0.3.1
+ restructured simplification pipeline: ranker and CWI use common scoring class
+ single-word requests always pass through CWI
+ featurizers still very POC, next step is to implement strong models

### Version 0.3
+ no more pickling!
+ POS-based synonym selection

### Version 0.2.5
+ more general database error handling

### Version 0.2.4
+ bugfix in Database connection: rollback connection at error

### Version 0.2.3
+ using synonym list for Danish
+ return simplification objects with unique sessionIds
+ accommodate for on-demand simplifications

### Version 0.2.2
+ simplify HTML only between given start and end character offset

### Version 0.2.1
+ blacklist words per user

### Version 0.2.0
+ massive restructuring of source
+ marking if original word is displayed as first alternative

### Version 0.1.2
+ log frontend_version in database

### Version 0.1.1
+ small bugfix in database calls

### Version 0.1
+ initial tagged release