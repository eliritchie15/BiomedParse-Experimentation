import os

def list_classes(filename):
    path = os.path.join("src", "model", filename)
    print(f"\n📂 SCANNING: {path}")
    
    if not os.path.exists(path):
        print("❌ File not found!")
        return

    try:
        with open(path, 'r', encoding='utf-8') as f:
            found_any = False
            for line in f:
                # Look for lines starting with "class "
                if line.strip().startswith("class "):
                    print(f"   found -> {line.strip()}")
                    found_any = True
            if not found_any:
                print("   (No classes found in this file)")
    except Exception as e:
        print(f"❌ Error reading file: {e}")

# Scan the two most likely suspects
# Scan the decoder folder to find the Predictor class
list_classes("transformer_decoder/mask2former_transformer_decoder.py")