# 🗣️🌿 Understanding Public Discourse on Home Decarbonisation 🌿🗣️

The `asf_public_discourse_home_decarbonisation` repository contains code to **analyse public discourse data from online forums** to identify:
- **Topics of conversation**
- **Frequently asked questions posted in the forum**
- **How the above change over time**

For the purpose of this project, the analyses are focused on conversations about **home heating** and **home decarbonisation**. However, *the pipelines created here can be applied to any domain area*.

In the context of the 🌿 **[sustainable future mission](https://www.nesta.org.uk/sustainable-future/)** 🌿 work, the analyses in this codebase allow us to:
- understand existing difficulties around installing low-carbon heating tech, such as heat pumps, and identify the skills/training needed by heating engineers to help them transition to low carbon heating;
- identify unknown issues faced by homeowners when installing low carbon heating tech in their home and questions they typically have;
- identify the common misconceptions around low-carbon heating tech;
- identify frequently asked questions by engineers installing low carbon heating tech;
- track the dominant narratives about heat pumps (and other heating technologies) across online forums.

You can read more about this project [here](https://www.nesta.org.uk/project/understanding-public-discourse-on-home-decarbonisation/).

## 🗂️ Repository structure
[to add]

## 🆕 Latest analyses
Latest data collection: [to add]
- When new data is available please open a new issue/PR and update the configs [here](https://github.com/nestauk/asf_public_discourse_home_decarbonisation/blob/dev/asf_public_discourse_home_decarbonisation/config/base.yaml).

## 🗞 Publications
[coming soon...]

## Setup
- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `direnv` and `conda`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure `pre-commit`
- Download and install the [Averta font](https://github.com/deblynprado/neon/blob/master/fonts/averta/Averta-Regular.ttf)
- Download spacy model: `python -m spacy download en_core_web_sm`

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
