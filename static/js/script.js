document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("uploadForm");
  const fileInput = document.querySelector('input[type="file"]');
  const loadingOverlay = document.getElementById("loadingOverlay");

  form.addEventListener("submit", (e) => {
    if (!fileInput.value) {
      e.preventDefault();
      alert("⚠️ Please select an image before analyzing!");
    } else {
      // Show the spinner overlay
      loadingOverlay.style.display = "block";
    }
  });
});
