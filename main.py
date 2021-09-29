from all_utils.model import model
from all_utils.utils import prepare_data, save_plot


model_clf, weights, biases = model()

X_valid, X_train, y_valid, y_train, X_test  = prepare_data("mnist")
EPOCHS = 30
VALIDATION = (X_valid, y_valid)

history = model_clf.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION)

save_plot(history,"history_plot")

model_clf.save("model.h5")