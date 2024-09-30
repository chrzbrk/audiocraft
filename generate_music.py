import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Lade das vortrainierte Modell (z.B. musicgen-medium)
model = MusicGen.get_pretrained('facebook/musicgen-medium')

# Setze die Generierungsparameter (z.B. 8 Sekunden Musik)
model.set_generation_params(duration=8)

# Generiere eine Musik basierend auf einer Textbeschreibung
descriptions = ['relaxing ambient music', 'energetic EDM', 'classical piano piece']
generated_audio = model.generate(descriptions)

# Speichere die generierten Audiodateien
for idx, audio in enumerate(generated_audio):
    audio_write(f'output_{idx}.wav', audio.cpu(), model.sample_rate)
