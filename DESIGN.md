---
name: MasterThesis
description: Native Streamlit research experiment workspace
colors:
  framework-primary: "rgb(255, 75, 75)"
  framework-primary-content: "rgb(255, 255, 255)"
typography:
  headline:
    fontFamily: "'Source Sans', sans-serif"
    fontSize: "44px"
    fontWeight: 700
    lineHeight: "52.8px"
  body:
    fontFamily: "'Source Sans', sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: "25.6px"
rounded:
  framework-control: "8px"
spacing:
  button-block: "4px"
  button-inline: "12px"
  navigation-inline: "8px"
components:
  button-primary:
    backgroundColor: "{colors.framework-primary}"
    textColor: "{colors.framework-primary-content}"
    typography: "{typography.body}"
    rounded: "{rounded.framework-control}"
    padding: "4px 12px"
  navigation-item:
    rounded: "{rounded.framework-control}"
    padding: "0px 8px"
---

# Design System: MasterThesis

## Overview

**Creative North Star: "Native research workspace"**

The interface uses Streamlit's existing visual language rather than a separate brand skin. Native headings, captions, controls, tables, alerts and disclosure widgets carry the hierarchy. This is a code-led system record, not an approved visual comp or a replacement visual world.

The framework owns rendering, focus treatment, theme switching and responsive behavior. The extracted frontmatter records the observed Streamlit 1.59.2 defaults; it is not an instruction to hard-code those values over the framework. Both light and dark themes remain supported. App code does not introduce custom CSS, decorative imagery or animation.

**Key Characteristics:**

- Native Streamlit widgets and theme ownership.
- Clear semantic headings and short explanatory captions.
- Fluid controls and data tables, with optional detail disclosed on demand.
- Turkish action labels with exact research identifiers and metric names retained.

Evidence: `app/streamlit_app.py`, `app/standard_launcher.py`, `app/experiment_ui.py`, and pages 0, 1, 3, 4, 8, 9 and 10. Browser evidence covers 1280 × 720 desktop and 390 × 844 mobile captures in `.impeccable/review/`. Computed primary-button, heading and navigation measurements were sampled on desktop. Theme-specific neutral colors and unmeasured component dimensions are deliberately not promoted to tokens.

## Colors

The accent and neutral surfaces belong to the active Streamlit theme, not to a project-specific palette.

### Primary

- **Framework Primary** (`colors.framework-primary`): the observed primary-button fill and border. Use `type="primary"` rather than applying its color manually.

### Neutral

- **Framework Primary Content** (`colors.framework-primary-content`): observed text on the primary button.
- Page, sidebar, field, text, divider and disabled colors follow the active theme. Dark screenshots are evidence of the selected runtime theme, not a dark-only product requirement.

**The Theme Ownership Rule.** Request semantic Streamlit variants; do not freeze theme-dependent neutrals into app styling.

## Typography

**Body Font:** the framework-provided Source Sans stack, recorded in frontmatter.

**Character:** familiar, utilitarian and readable. Hierarchy comes from `st.title`, `st.header`, `st.subheader`, ordinary labels and `st.caption`, not custom display typography.

### Hierarchy

- **Headline** (`typography.headline`): observed `st.title` styling on desktop.
- **Body** (`typography.body`): observed primary-button text styling; the navigation also uses the same family, size and weight, with its own framework line height.
- **Captions and technical code:** use `st.caption` and `st.code` without overriding the framework's sizes or font choices. Their exact values were not sampled.

The measured title has framework padding of 20px above and 16px below; this is evidence, not a new project spacing scale. No custom type scale is defined.

## Layout

The entry point uses Streamlit's wide layout and a native sidebar. Forms use two related-selection columns and three common-training columns where appropriate. The framework stacks these columns on narrow screens; mobile captures confirm the dataset/model controls stack and the sidebar becomes a drawer.

Use stretch-width native controls and dataframes where the source already does. Keep native page flow and vertical rhythm; there are no project breakpoints, fixed-width custom elements or custom grid tokens. Navigation and first-viewport strategy are defined in `.impeccable/surfaces/experiment-workspace.md`, not as universal visual-system constraints here.

## Elevation & Depth

No app-authored shadow or elevation vocabulary exists. The interface relies on the active theme's surface contrast, borders and widget states. Framework overlays and focus treatments remain framework-owned; do not replace them with a custom shadow system. There is no app-authored motion grammar.

## Shapes

Native controls use the gently rounded framework form language. The observed primary button and navigation item share `rounded.framework-control`; this does not establish a new global radius override. Inputs, expanders and dataframes retain their own native borders and shapes. Do not manufacture cards or badges merely to reuse a radius.

## Components

### Buttons

Native and explicit. Primary actions use `st.button(..., type="primary")`; routine actions use the default variant. The primary button's observed radius, padding and typography are recorded in frontmatter. Its observed border is 1px solid in the primary fill color. Hover, focus, disabled and active states remain Streamlit-owned; measured default colors do not substitute for those states.

### Inputs / Fields

Use native selectboxes, number inputs, text inputs, checkboxes and radios. Preserve visible labels, native keyboard behavior and disabled-state semantics. Do not hide validation errors inside collapsed disclosures. No input-specific color, radius, padding or focus token was measured for this record.

### Navigation

The source uses `st.page_link` inside `st.sidebar`, with a stretch-width primary action above the links. The sampled link uses the documented navigation padding and radius, a 32px line height and an 8px internal gap. Selection, hover, focus and mobile drawer behavior are supplied by Streamlit, including their theme-dependent colors.

### Disclosure

`st.expander` groups advanced settings, commands, logs and technical metadata. The native summary remains a visible, keyboard-operable control. Warnings and errors that prevent progress stay visible rather than being hidden by default.

### Tables and Research Figures

Use `st.dataframe(..., hide_index=True, width="stretch")` for result tables. Model identity and available scores lead the compact table; the original table remains accessible in details. Native charts and existing scientific report images are data, not decorative imagery. Do not introduce a custom card or chart skin.

The sidecar contains small self-contained semantic previews for the design panel. They are not runtime replacements for Streamlit widgets, do not reproduce unmeasured theme internals, and must not be copied into the app as injected HTML/CSS.

## Do's and Don'ts

### Do:

- **Do** use native Streamlit widgets and semantic variants.
- **Do** preserve the selectable light and dark themes and native focus behavior.
- **Do** use semantic headings, visible labels and concise captions.
- **Do** disclose optional technical detail while keeping blocking errors visible.
- **Do** retain original research metric names and existing safety confirmations.

### Don't:

- **Don't** inject custom CSS or replace native widgets with sidecar previews.
- **Don't** turn the captured dark theme into a fixed dark-only palette.
- **Don't** add decorative cards, gradients, emoji navigation, imagery or motion.
- **Don't** treat sampled defaults or synthetic panel previews as a new project theme.
