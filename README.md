# maj-biostat.github.io

## Create new post

Copy an existing post out of the notebooks folder then remove the contents bar the yaml and start with the entry.
During development, just render the page you are working on (or use preview):

```
quarto render notebooks/test.qmd --to html
```

Note, I have 

```
execute:
  freeze: true
```

in the _quarto.yml file. 

This means never re-render during project render and so if you need the page/document to be re-rendered, you will have to render that document individually.
Bascially, I want to avoid re-rendering old posts unless I explicitly need to do so.

You need to include the pages explicitly in the `index.qmd` file!!!!

Once all the pages are rendered, commit and push the content (including the docs folder which contains the rendered site).

The post will be picked up by the listing setup in the main landing page (`index.qmd`).


