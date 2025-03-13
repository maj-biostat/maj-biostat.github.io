# maj-biostat.github.io

## Create new post

Copy an existing post out of the notebooks folder then remove the contents bar the yaml and start with the entry.
During development, just render the page you are working on:

```
quarto render notebooks/test.qmd --to html
```

then review the html file created under the docs director in a browser.

Once happy, render the site (it will take a while)

```
quarto render
```

then commit and push the content (including the docs folder which contains the rendered site).

The post will be picked up by the listing setup in the main landing page (`index.qmd`).


