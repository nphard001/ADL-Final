# --------==== main Makefile in ADL final project ====--------
all:
	@echo nothing to build
# ================================================================
chatbot_start:
	cd chatbot && $(UWSGI) --ini="uwsgi.ini"
chatbot_start_by_django:
	cd chatbot && python manage.py start
chatbot_touch_ini:
	touch chatbot/uwsgi.ini

fr_train_sl:
	cd fashion_retrieval && python train_sl.py --log-interval=50 --lr=0.001  --batch-size=128 --model-folder="models/"
frm_train_sl:
	python -m fashion_retrieval.train_sl --log-interval=50 --lr=0.001  --batch-size=128 --model-folder="models/"

