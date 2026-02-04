1. Generate Dev Models (for local testing)

cd scripts/models

# Generate models with random weights
./convert_blazeface.py -o blazeface.safetensors
./convert_eye_state.py --random-init -o eye_state.safetensors
./convert_u2net.py --random-init -o u2net.safetensors

# Install to local cache
mkdir -p ~/Library/Application\ Support/photo-qa/models
cp *.safetensors ~/Library/Application\ Support/photo-qa/models/

# Verify
cargo run -- models list

2. Obtain Production Weights

BlazeFace — needs training (our arch differs from reference):
- Train on http://shuoyang1213.me/WIDERFACE/ dataset
- Or adapt MediaPipe's TFLite weights (requires architecture adjustment)

Eye State — train on CEW dataset:
# Download CEW: http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html
# Organize as: cew/{closed_eyes,open_eyes}/*.jpg
./convert_eye_state.py --data-dir /path/to/cew -o eye_state.safetensors

U²-Net — download official weights:
# Download from Google Drive:
# https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view
./convert_u2net.py --weights u2net_pretrained.pth -o u2net.safetensors

3. Verify & Get Checksums

./verify_models.py --blazeface blazeface.safetensors \
                    --eye-state eye_state.safetensors \
                    --u2net u2net.safetensors

sha256sum *.safetensors

4. Update models.rs

Edit crates/photo-qa-adapters/src/models.rs — replace placeholder hashes:
sha256: "<real-hash-here>",

5. Create GitHub Release

./release.sh

6. Test Auto-Download

rm -rf ~/Library/Application\ Support/photo-qa/models
cargo run -- models fetch
cargo run -- check /path/to/test.jpg

---
Shortcut for now: Ship with random-init models for v0.1.0-alpha, add trained models in v0.2.0. The placeholder checksums already skip
verification.