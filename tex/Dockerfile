FROM registry.gitlab.com/novanext/base-docker:master

# collection for a package can be found by searching, e.g. for "depend siunitx"
# http://ctan.triasinformatica.nl/systems/texlive/tlnet/tlpkg/texlive.tlpdb
# ADD ./texlive.profile /texlive.profile

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update\
    && apt-get install -y wget perl make libfontconfig