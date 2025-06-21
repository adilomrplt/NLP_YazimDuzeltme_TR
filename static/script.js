const input = document.getElementById("metin");
const oneriDiv = document.getElementById("canli_oneri");

input.addEventListener('input', function() {
    let kelime = this.value.split(/\s+/).pop();
    if (kelime.length > 0) {
        fetch('/duzelt', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({kelime: kelime})
        })
        .then(r => r.json())
        .then(data => {
            if (data.tahmin && data.tahmin !== kelime) {
                oneriDiv.style.display = "block";
                oneriDiv.innerHTML = `<b>Bunu mu demek istediniz?</b> <span style="color:#D7263D">${data.tahmin}</span>`;
            } else {
                oneriDiv.style.display = "none";
            }
        });
    } else {
        oneriDiv.style.display = "none";
    }
});
