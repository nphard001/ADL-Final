# ssh specific make
HOST = linux7
HOST_PREFIX = /tmp2/b04303128
PROJECT = ADL-Final
HOST_PROJECT := $(HOST_PREFIX)/$(PROJECT)/
MINI_DIR = $(HOST_PREFIX)/mini
MINI_BIN = $(HOST_PREFIX)/mini/bin
CONDA := $(MINI_BIN)/conda
PIP := $(MINI_BIN)/pip
PYTHON := $(MINI_BIN)/python
UWSGI := $(MINI_BIN)/uwsgi
RSYNC_EXCLUDE_LIST = *.git* *__pycache__* *.sqlite3* *static/*
RSYNC_EXCLUDE := $(RSYNC_EXCLUDE_LIST:%=--exclude="%")
linux7:
	$(MAKE) linux7_routine
linux7_routine:
	rsync -avzh ../$(PROJECT)/ $(HOST):$(HOST_PROJECT) $(RSYNC_EXCLUDE)
	ssh linux7 "cd $(HOST_PROJECT) && make chatbot_touch_ini"
linux7_migrate:
	rsync -avzh ../$(PROJECT)/ $(HOST):$(HOST_PROJECT) $(RSYNC_EXCLUDE)
	ssh linux7 "cd $(HOST_PROJECT) && make migrate_host_data && make chatbot_touch_ini"
linux7_full:
	rsync -avzh ../$(PROJECT)/ $(HOST):$(HOST_PROJECT) $(RSYNC_EXCLUDE)
linux7_static:
	rsync -avzh ../$(PROJECT)/static/ $(HOST):$(HOST_PROJECT)static/
linux7_demo:
	rsync -avzh ../$(PROJECT)/static/demo/ $(HOST):$(HOST_PROJECT)static/demo/

FORWARDER = /Users/qtwu/Drive/Active/forwarder
forwarder_auto forwarder_sync_adl forwarder_sync_heroku forwarder_hang_heroku_log: forwarder_%:
	make $* -C $(FORWARDER)
