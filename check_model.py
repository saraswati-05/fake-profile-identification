import joblib

# Use full path with raw string
model = joblib.load(r"E:\Saraswati Sonale\fake profile identification project\Fake social profile detection\ml\model.pkl")

print("✅ Model loaded successfully!")
print("Model type:", type(model))

if hasattr(model, "get_params"):
    print("\nModel Parameters:\n", model.get_params())

if hasattr(model, "feature_names_in_"):
    print("\nExpected Features:\n", model.feature_names_in_)

if hasattr(model, "n_features_in_"):
    print("\nNumber of Features Expected:", model.n_features_in_)
