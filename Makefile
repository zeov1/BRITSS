MODEL=rits_i
EPOCHS=1000
BATCH_SIZE=1000
IMPUTE_WEIGHT=0.3
LABEL_WEIGHT=1.0
HID_SIZE=108

all: run

.PHONY: run
run:
	python main.py 	--model $(MODEL) \
					--epochs $(EPOCHS) \
					--batch_size $(BATCH_SIZE) \
					--impute_weight $(IMPUTE_WEIGHT) \
					--label_weight $(LABEL_WEIGHT) \
					--hid_size $(HID_SIZE)