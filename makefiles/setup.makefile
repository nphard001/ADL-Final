# setup.makefile: things you won't run them twice

# for "fashion-retrieval":
# just download original git repo, instead of fork
setup_duplicate:
	git clone https://github.com/nphard001/fashion_retrieval
	rm -rf fashion_retrieval/.git/
	tree -d .
	@echo "NOTES: original .git files are deleted"

setup_touch_init:
	touch fashion_retrieval/__init__.py
	touch fashion_retrieval/captioner/__init__.py
	touch fashion_retrieval/captioner/neuraltalk2/__init__.py

setup_symlink_chatbot:
	test ls chatbot/nphard001 || ln -s ../nphard001/ chatbot/nphard001
	test ls fashion_retrieval/nphard001/ || ln -s ../nphard001/ fashion_retrieval/nphard001/
	test ls fashion_retrieval_old/nphard001/ || ln -s ../nphard001/ fashion_retrieval_old/nphard001/
	test ls chatbot/fashion_retrieval/ || ln -s ../fashion_retrieval/ chatbot/fashion_retrieval/
	test ls chatbot/fashion_retrieval_old/ || ln -s ../fashion_retrieval_old/ chatbot/fashion_retrieval_old/
	tree -d chatbot
	
setup_symlink_static_model_fashion_retrieval_old: setup_symlink_static_model_%:
	-ln -s ../../static/model/rl-13.pt $*/models/rl-13.pt
	-ln -s ../../static/model/features/256embedding.p $*/features/256embedding.p
	-ln -s ../../static/model/features/att_feature.npz $*/features/att_feature.npz
	-ln -s ../../static/model/features/fc_feature.npz $*/features/fc_feature.npz
	-ln -s ../../static/model/caption_models/infos_best.pkl $*/caption_models/infos_best.pkl
	-ln -s ../../static/model/caption_models/model_best.pth $*/caption_models/model_best.pth
	tree $*

install_chatbot:
	$(PIP) install line-bot-sdk
	$(PIP) install Pillow
	$(PIP) install sklearn
