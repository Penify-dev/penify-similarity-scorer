# Semantischer Ähnlichkeits-Scorer

Ein FastAPI-Server, der die semantische Ähnlichkeit zwischen zwei Zeichenketten mit einem sentence-transformers-Modell vergleicht.

## Funktionen

- Semantischer Ähnlichkeitsvergleich zwischen Textzeichenketten
- Verwendet sentence-transformers-Modell (standardmäßig all-MiniLM-L6-v2)
- Nutzt automatisch GPU, falls verfügbar
- Einfache REST-API mit JSON-Anfrage/Antwort
- Modell-Caching, um wiederholte Downloads bei Server-Neustart zu vermeiden

## Installation

1. Klonen Sie dieses Repository:
```bash
git clone https://github.com/yourusername/similarity-scorer.git
cd similarity-scorer
```

2. Installieren Sie die erforderlichen Pakete:
```bash
pip install -r requirements.txt
```

## Verwendung

### Option 1: Direkt mit Python ausführen

#### Entwicklungsmodus
```bash
uvicorn main:app --reload
```

#### Produktionsmodus mit Gunicorn
```bash
./start.sh
```

Sie können Einstellungen anpassen, indem Sie Umgebungsvariablen setzen:
```bash
MODEL_NAME=all-mpnet-base-v2 WORKERS=4 ./start.sh
```

Der Server läuft unter: `http://127.0.0.1:16000`

Interaktive API-Dokumentation ist verfügbar unter: `http://127.0.0.1:16000/docs`

### Option 2: Mit Docker ausführen

```bash
docker build -t similarity-scorer .
docker run -p 16000:16000 similarity-scorer
```

### Option 3: Mit Docker Compose ausführen

```bash
docker-compose up
```

Um im Detached-Modus zu laufen:
```bash
docker-compose up -d
```

Um den Dienst zu stoppen:
```bash
docker-compose down
```

## Server-Verwaltung

### Server starten

Um den Server im Produktionsmodus zu starten:

```bash
./start.sh
```

Für speicherkonfigurierte Umgebungen:

```bash
./start_optimized.sh
```

### Server stoppen

Um alle laufenden Server-Instanzen zu stoppen:

```bash
./terminate.sh
```

Wenn die reguläre Terminierung nicht funktioniert, verwenden Sie das Force-Terminierung-Skript (kann sudo erfordern):

```bash
sudo ./force_terminate.sh
```

Diese Skripte werden:

1. Alle laufenden Prozesse identifizieren, die mit dem Similarity-Scorer zusammenhängen
2. Versuchen, sie ordnungsgemäß zu terminieren (reguläres Skript) oder zwangsweise (Force-Skript)
3. Den Status nach dem Terminierungsversuch melden

## API-Endpunkte

### POST /compare

Vergleicht die semantische Ähnlichkeit zwischen zwei Sätzen.

**Anfrage-Body:**
```json
{
  "sentence1": "Wie backe ich einen Kuchen?",
  "sentence2": "Was ist der Prozess zum Herstellen eines Kuchens?"
}
```

**Antwort:**
```json
{
  "sentence1": "Wie backe ich einen Kuchen?",
  "sentence2": "Was ist der Prozess zum Herstellen eines Kuchens?",
  "semantic_similarity": 0.87
}
```

## Anpassung

### Modell ändern

Sie können das Modell auf zwei Arten ändern:

1. Durch Setzen der `MODEL_NAME` Umgebungsvariablen:

   ```bash
   # Beim Ausführen mit Python
   MODEL_NAME=all-mpnet-base-v2 uvicorn main:app --reload
   
   # Beim Ausführen mit Docker
   docker run -p 16000:16000 -e MODEL_NAME=all-mpnet-base-v2 similarity-scorer
   
   # Oder aktualisieren Sie die Umgebungsvariable in docker-compose.yml
   # und führen Sie dann docker-compose up aus
   ```

2. Durch direktes Bearbeiten der Standardeinstellung in `main.py`:

   ```python
   model_name = os.environ.get("MODEL_NAME", "all-mpnet-base-v2")
   ```

Beachten Sie, dass genauere Modelle wie "all-mpnet-base-v2" mehr Rechenressourcen benötigen, aber bessere Ähnlichkeitsergebnisse liefern.

## Leistungsüberlegungen

### Modell-Caching

Die Anwendung ist so konfiguriert, dass heruntergeladene Modelle im `models/` Verzeichnis gecacht werden. Das bedeutet:

1. Das Modell wird nur einmal heruntergeladen, auch wenn Sie den Server mehrmals neu starten
2. Nachfolgende Server-Starts sind viel schneller
3. Bei Verwendung von Docker wird der Modell-Cache in einem benannten Volume für Persistenz gespeichert

Dies ist besonders wichtig für größere Modelle wie "all-mpnet-base-v2", die mehrere hundert MB groß sein können.

### Gemeinsamer Modell-Prozess

Um den Speicherverbrauch zu optimieren, verwendet die Anwendung einen dedizierten Modell-Service, der in einem separaten Prozess läuft:

1. Nur eine Kopie des Modells wird im Speicher geladen, unabhängig davon, wie viele Gunicorn-Worker laufen
2. Alle Worker-Prozesse kommunizieren mit dem Modell-Service über IPC (Inter-Prozess-Kommunikation)
3. Dies reduziert den Speicherverbrauch erheblich, wenn mit mehreren Workern gearbeitet wird

Diese Architektur ist besonders vorteilhaft für große Modelle, die sonst mehrere GB RAM verbrauchen würden, wenn sie separat in jedem Worker-Prozess geladen würden.

## Speicheroptimierung

Diese Anwendung ist für Speichereffizienz konzipiert, insbesondere in Umgebungen mit begrenzten Ressourcen:

1. **Einzelne Modell-Instanz**: Das Modell wird nur einmal in einem dedizierten Prozess geladen und über alle Worker geteilt
2. **Konservative Worker-Anzahl**: Die Gunicorn-Konfiguration verwendet weniger Worker als typisch, um den Speicherverbrauch zu reduzieren
3. **Speicherüberwachung**: Die Anwendung protokolliert den Speicherverbrauch an verschiedenen Punkten, um die Ressourcennutzung zu verfolgen
4. **Konfigurierbare Worker-Anzahl**: Sie können die `WORKERS` Umgebungsvariable setzen, um Worker weiter zu begrenzen
5. **Worker-Lebenszyklus-Verwaltung**: Worker werden nach der Bearbeitung einer bestimmten Anzahl von Anfragen neu gestartet, um Speicherlecks zu verhindern

### Optimierter Start

Für Umgebungen mit sehr begrenztem Speicher verwenden Sie das optimierte Startskript:

```bash
./start_optimized.sh
```

Dieses Skript setzt Umgebungsvariablen zur Optimierung des Speicherverbrauchs und bietet Speicherverbrauchsverfolgung.

### GPU-Unterstützung

Die Anwendung erkennt automatisch verfügbare GPU-Ressourcen und nutzt sie:

#### macOS GPU-Unterstützung (Apple Silicon/AMD)

Auf macOS verwendet die Anwendung:
- Apple's Metal Performance Shaders (MPS) Backend auf Apple Silicon (M1/M2/M3) Macs
- AMD GPUs auf Intel Macs mit kompatiblen Grafikkarten

Um zu überprüfen, ob Ihr Mac GPU-Beschleunigung verwendet:
1. Starten Sie den Server
2. Besuchen Sie `http://127.0.0.1:16000/system-info`, um Geräteinformationen zu sehen
3. Suchen Sie nach `"mps_available": true` und `"current_device": "mps"` in der Antwort

#### NVIDIA GPU-Unterstützung (Linux/Windows)

Auf Systemen mit NVIDIA GPUs wird CUDA automatisch verwendet.

Wenn Sie in Docker laufen, stellen Sie sicher, dass die GPU-Runtime-Konfiguration wie in der docker-compose.yml-Datei angegeben enthalten ist.

## Lizenz

[MIT License](LICENSE)