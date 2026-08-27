// Language switcher
document.getElementById('lang-switcher')?.addEventListener('change', function() {
    fetch('/api/language', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({language: this.value})
    }).then(() => location.reload());
});

// Audio prompt playback
function playAudio(key, lang) {
    const audio = document.getElementById('voice-prompt');
    const path = `/static/../audio/${lang}/${key.replace(/\./g, '_')}.mp3`;
    audio.src = path;
    audio.play().catch(() => {});
}

// Auto-play voice prompt on quality feedback page if retake
const autoPlayKey = document.querySelector('[data-audio-key]');
if (autoPlayKey) {
    const key = autoPlayKey.dataset.audioKey;
    const lang = autoPlayKey.dataset.audioLang || 'hi';
    playAudio(key, lang);
}

// Eye selector toggle
document.querySelectorAll('.eye-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.eye-btn').forEach(b => b.classList.remove('selected'));
        this.classList.add('selected');
        document.getElementById('eye-input').value = this.dataset.eye;
    });
});

// File upload preview
const fileInput = document.getElementById('image-upload');
if (fileInput) {
    fileInput.addEventListener('change', function() {
        const preview = document.getElementById('upload-preview');
        if (this.files && this.files[0] && preview) {
            const reader = new FileReader();
            reader.onload = e => {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(this.files[0]);
        }
    });
}
