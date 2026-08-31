/**
 * NETRA-AI — app.js
 * Visual-first. No audio dependency.
 */

'use strict';

/* ── Language switcher ────────────────────────────────── */
(function () {
  const sel = document.getElementById('lang-switcher');
  if (!sel) return;
  sel.addEventListener('change', function () {
    fetch('/api/language', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lang: this.value }),
    }).then(function (r) {
      if (r.ok) location.reload();
    }).catch(function () {
      location.reload(); // graceful fallback
    });
  });
})();

/* ── Eye selector toggle ──────────────────────────────── */
(function () {
  const buttons = document.querySelectorAll('.eye-btn');
  const hidden  = document.getElementById('eye-input');
  if (!buttons.length || !hidden) return;

  buttons.forEach(function (btn) {
    btn.addEventListener('click', function () {
      buttons.forEach(function (b) {
        b.classList.remove('selected');
        b.setAttribute('aria-pressed', 'false');
      });
      btn.classList.add('selected');
      btn.setAttribute('aria-pressed', 'true');
      hidden.value = btn.dataset.eye;
    });

    // Keyboard support
    btn.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        btn.click();
      }
    });
  });
})();

/* ── Camera file upload → preview ────────────────────── */
(function () {
  function wireInput(inputId, previewWrapId, previewId) {
    const inp  = document.getElementById(inputId);
    const wrap = document.getElementById(previewWrapId);
    const img  = document.getElementById(previewId);
    if (!inp) return;

    inp.addEventListener('change', function () {
      const file = this.files && this.files[0];
      if (!file) return;
      if (file.size > 10 * 1024 * 1024) {
        showFormError('Image too large (max 10 MB). Please compress and retry.');
        inp.value = '';
        return;
      }
      const reader = new FileReader();
      reader.onload = function (e) {
        if (wrap) wrap.classList.add('visible');
        if (img)  img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    });
  }

  wireInput('image-upload', 'preview-wrap', 'upload-preview');
})();

/* ── Score bars — animate on load ────────────────────── */
(function () {
  // Bars start at width:0 via inline style, CSS transition does the rest.
  // Nothing extra needed; the width is already set inline in the template.
})();

/* ── Confidence fill — animate on load ───────────────── */
(function () {
  const fill = document.querySelector('.confidence-fill');
  if (!fill) return;
  const target = fill.style.width;
  fill.style.width = '0%';
  requestAnimationFrame(function () {
    requestAnimationFrame(function () {
      fill.style.width = target;
    });
  });
})();

/* ── History: keyboard-accessible clickable rows ─────── */
(function () {
  document.querySelectorAll('.history-table tr.clickable').forEach(function (row) {
    row.setAttribute('tabindex', '0');
  });
})();

/* ── Helpers ──────────────────────────────────────────── */
function showFormError(msg) {
  const el = document.getElementById('form-error');
  if (!el) return;
  el.textContent = msg;
  el.style.display = 'flex';
  el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
