########################################################################
#
# Makefile for MLHub package
#
# Time-stamp: <Friday 2021-04-16 12:10:19 AEST Graham Williams>
#
# Copyright (c) Graham.Williams@togaware.com
#
# License: Creative Commons Attribution-ShareAlike 4.0 International.
#
########################################################################

# App version numbers
#   Major release
#   Minor update
#   Trivial update or bug fix

APP=zynlp
VER=0.0.1
DATE=$(shell date +%Y-%m-%d)

########################################################################
# Supported modules.

INC_BASE    = $(HOME)/.local/share/make
INC_CLEAN   = $(INC_BASE)/clean.mk
#NC_R       = $(INC_BASE)/r.mk
#NC_KNITR   = $(INC_BASE)/knitr.mk
#NC_PANDOC  = $(INC_BASE)/pandoc.mk
INC_GIT     = $(INC_BASE)/git.mk
#NC_AZURE   = $(INC_BASE)/azure.mk
#NC_LATEX   = $(INC_BASE)/latex.mk
#NC_PDF     = $(INC_BASE)/pdf.mk
#NC_DOCKER  = $(INC_BASE)/docker.mk
INC_MLHUB   = $(INC_BASE)/mlhub.mk
#NC_WEBCAM  = $(INC_BASE)/webcam.mk

ifneq ("$(wildcard $(INC_CLEAN))","")
  include $(INC_CLEAN)
endif
ifneq ("$(wildcard $(INC_R))","")
  include $(INC_R)
endif
ifneq ("$(wildcard $(INC_KNITR))","")
  include $(INC_KNITR)
endif
ifneq ("$(wildcard $(INC_PANDOC))","")
  include $(INC_PANDOC)
endif
ifneq ("$(wildcard $(INC_GIT))","")
  include $(INC_GIT)
endif
ifneq ("$(wildcard $(INC_AZURE))","")
  include $(INC_AZURE)
endif
ifneq ("$(wildcard $(INC_LATEX))","")
  include $(INC_LATEX)
endif
ifneq ("$(wildcard $(INC_PDF))","")
  include $(INC_PDF)
endif
ifneq ("$(wildcard $(INC_DOCKER))","")
  include $(INC_DOCKER)
endif
ifneq ("$(wildcard $(INC_MLHUB))","")
  include $(INC_MLHUB)
endif
ifneq ("$(wildcard $(INC_WEBCAM))","")
  include $(INC_WEBCAM)
endif

define HELP
$(APP):

  install	Install into local pre-existing .mlhub folder.

endef
export HELP

help::
	@echo "$$HELP"

install:
	install mlhub/demo.py mlhub/sentiment.py ~/.mlhub/zynlp/
