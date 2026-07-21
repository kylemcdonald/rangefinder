#!/bin/bash
# Download additional public benchmark controls (all CC-BY-4.0, Zenodo).
# Provenance and licenses are recorded in controls/README.md.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p controls/extra tmp/downloads

log() { echo "[$(date +%H:%M:%S)] $*"; }

fetch() { # fetch <url> <dest>
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then log "SKIP (exists): $dest"; return 0; fi
  log "GET $url -> $dest"
  curl -sL --fail --retry 3 -o "$dest.part" "$url" && mv "$dest.part" "$dest" \
    || { log "FAILED: $url"; rm -f "$dest.part"; return 1; }
}

# Record metadata for provenance
for rec in 5534859 10677562 14848236 14221628 15446699; do
  fetch "https://zenodo.org/api/records/$rec" "tmp/downloads/zenodo_$rec.json"
done

# 1. H/D-charged steel: shared RRNG + the smallest POS files (fast smoke tests)
python3 - <<'EOF'
import json, subprocess, pathlib
rec = json.load(open('tmp/downloads/zenodo_5534859.json'))
files = rec.get('files', [])
rrng = [f for f in files if f['key'].lower().endswith('.rrng')]
poses = sorted([f for f in files if f['key'].lower().endswith('.pos')], key=lambda f: f['size'])[:3]
for f in rrng + poses:
    name = f['key'].replace(' ', '_')
    dest = pathlib.Path('controls/extra') / f"steelHD_5534859_{name}"
    if dest.exists() and dest.stat().st_size == f['size']:
        print('SKIP', dest); continue
    url = f"https://zenodo.org/api/records/5534859/files/{f['key'].replace(' ', '%20')}/content"
    print('GET', url, '->', dest, f"({f['size']/1e6:.1f} MB)")
    subprocess.run(['curl', '-sL', '--fail', '--retry', '3', '-o', str(dest), url], check=False)
EOF

# 2. Fe-oxide stoichiometry standards: take the smallest of the three epos
python3 - <<'EOF'
import json, subprocess, pathlib
rec = json.load(open('tmp/downloads/zenodo_10677562.json'))
files = sorted([f for f in rec.get('files', []) if f['key'].lower().endswith('.epos')], key=lambda f: f['size'])
meta = {f['key']: f['size'] for f in files}
print('available:', meta)
print('description snippet:', rec.get('metadata', {}).get('description', '')[:2000])
for f in files[:1]:
    dest = pathlib.Path('controls/extra') / f"feoxide_10677562_{f['key'].replace(' ', '_')}"
    if dest.exists() and dest.stat().st_size == f['size']:
        print('SKIP', dest); continue
    url = f"https://zenodo.org/api/records/10677562/files/{f['key'].replace(' ', '%20')}/content"
    print('GET', url, '->', dest, f"({f['size']/1e6:.1f} MB)")
    subprocess.run(['curl', '-sL', '--fail', '--retry', '3', '-o', str(dest), url], check=False)
EOF

# 3. Pure Li epos
fetch "https://zenodo.org/api/records/14848236/files/R5076_68722.epos/content" \
      "controls/extra/li_14848236_R5076_68722.epos"

# 4. MassBank curated peak-list library (non-APT control source)
fetch "https://zenodo.org/api/records/14221628/files/MassBank-data-2024.11.zip/content" \
      "tmp/downloads/MassBank-data-2024.11.zip"

# 5. ToF-SIMS open text spectra (negative polarity TSB medium + one positive)
python3 - <<'EOF'
import json, subprocess, pathlib
rec = json.load(open('tmp/downloads/zenodo_15446699.json'))
files = [f for f in rec.get('files', []) if f['key'].lower().endswith('.txt')]
print('available:', [(f['key'], round(f['size']/1e6,1)) for f in files])
picks = [f for f in files if 'tsb' in f['key'].lower() or 'neg' in f['key'].lower()][:2] or files[:2]
for f in picks:
    dest = pathlib.Path('controls/extra') / f"tofsims_15446699_{f['key'].replace(' ', '_')}"
    if dest.exists() and dest.stat().st_size == f['size']:
        print('SKIP', dest); continue
    url = f"https://zenodo.org/api/records/15446699/files/{f['key'].replace(' ', '%20')}/content"
    print('GET', url, '->', dest, f"({f['size']/1e6:.1f} MB)")
    subprocess.run(['curl', '-sL', '--fail', '--retry', '3', '-o', str(dest), url], check=False)
EOF

# 6. (V,Al)N nitride film (plan-view, 800 C): .epos + expert .rrng in a zip.
#    New chemistry class (transition-metal nitride); range-file composition truth.
fetch "https://zenodo.org/records/7788883/files/VAlN_film_plan-view_800C.zip?download=1" \
      "tmp/downloads/vain_800C_7788883.zip"
if [ -s "tmp/downloads/vain_800C_7788883.zip" ] \
   && [ ! -s "controls/extra/vain_planview_800c_7788883.epos" ]; then
  log "extracting (V,Al)N nitride epos + rrng"
  unzip -o -j tmp/downloads/vain_800C_7788883.zip "*.epos" "*.rrng" -d tmp/downloads/vain_extract >/dev/null 2>&1
  cp tmp/downloads/vain_extract/VAlN_film_plan-view_800C.epos controls/extra/vain_planview_800c_7788883.epos
  cp tmp/downloads/vain_extract/VAlN_film_plan-view_800C.rrng controls/extra/vain_planview_800c_7788883.rrng
fi

log "All downloads attempted. Contents of controls/extra:"
ls -la controls/extra/
