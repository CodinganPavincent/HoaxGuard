document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('detect-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.querySelector('.btn-text');
    const loader = document.getElementById('loader');
    const resultContainer = document.getElementById('result-container');
    const verdictBadge = document.getElementById('verdict-badge');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const confidenceText = document.getElementById('confidence-text');
    
    // Result elements
    const resPublisher = document.getElementById('res-publisher');
    const resTitle = document.getElementById('res-title');
    const resContent = document.getElementById('res-content');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const urlStr = document.getElementById('news-url').value;
        if (!urlStr.trim()) {
            alert('Mohon masukkan URL tautan berita.');
            return;
        }

        // UI Loading State
        submitBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        
        // Reset Progress Bar
        progressBarFill.style.width = '0%';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url: urlStr })
            });

            const data = await response.json();

            if (!response.ok) {
                alert(data.error || 'Terjadi kesalahan pada server.');
                return;
            }

            // Update UI with Results
            updateResults(data);

        } catch (error) {
            alert('Gagal terhubung ke server. Pastikan server Flask sedang berjalan.');
            console.error(error);
        } finally {
            // Restore UI State
            submitBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });

    function updateResults(data) {
        resultContainer.classList.remove('hidden');
        
        const isHoax = data.prediction === 'HOAX';
        const confidencePct = Math.round(data.confidence * 100);

        // Update Badge
        verdictBadge.textContent = data.prediction;
        verdictBadge.className = 'badge ' + (isHoax ? 'hoax' : 'fact');

        // Update Progress Bar
        confidenceText.textContent = `${confidencePct}%`;
        
        // Update Extracted Info
        resPublisher.textContent = data.extracted_publisher || 'Tidak diketahui';
        resTitle.textContent = data.extracted_title || 'Tidak ditemukan';
        resContent.textContent = data.extracted_text || 'Tidak ada teks yang dapat diekstrak';
        
        // Update Search References
        const searchRefSection = document.getElementById('search-references-section');
        const refContainer = document.getElementById('references-container');
        refContainer.innerHTML = ''; // clear previous
        
        if (data.references && data.references.length > 0) {
            searchRefSection.classList.remove('hidden');
            data.references.forEach(ref => {
                const card = document.createElement('div');
                card.className = 'reference-card';
                card.innerHTML = `
                    <a href="${ref.url}" target="_blank" rel="noopener noreferrer">${ref.title}</a>
                    <p>${ref.snippet}</p>
                    <div class="reference-url">${ref.url}</div>
                `;
                refContainer.appendChild(card);
            });
        } else {
            searchRefSection.classList.add('hidden');
        }

        // Small delay to allow CSS transition to work
        setTimeout(() => {
            progressBarFill.style.width = `${confidencePct}%`;
            
            if (isHoax) {
                progressBarFill.style.backgroundColor = 'var(--danger-color)';
            } else {
                progressBarFill.style.backgroundColor = 'var(--success-color)';
            }
        }, 100);
    }
});
