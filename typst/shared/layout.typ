// shared/layout.typ
// Global layout and article-style template for MSL theory documents.

// Environment-like wrapper for an article-style document.
#let msl-article(
  title: none,
  subtitle: none,
  authors: (),
  date: auto,
  doc,
) = {
  set page(
    paper: "a4",
    margin: (top: 25mm, bottom: 25mm, x: 25mm),
  )

  set text(
    font: "Latin Modern Roman",
    size: 11pt,
  )

  set heading(
    numbering: "1.",
  )

  // Title block
  if title != none {
    align(center, [
      #if subtitle != none [
        #strong(title) \
        #emph(subtitle)
      ] else [
        #strong(title)
      ]
      #v(6mm)
      #if authors != () [
        #authors.join(", ") \
      ]
      #v(3mm)
      #if date != none [
        #text(0.9em, date)
      ]
      #v(10mm)
    ])
  }

  // Main content
  doc
}

// Convenience show-rule: wrap whole document in msl-article.
#let msl-article-default(
  title: none,
  subtitle: none,
  authors: (),
  date: auto,
  body,
) = {
  msl-article(
    title: title,
    subtitle: subtitle,
    authors: authors,
    date: date,
    body,
  )
}
