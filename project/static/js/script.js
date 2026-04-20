function confirmDelete() {
    return confirm("Do you really want to delete this student? This action cannot be undone.");
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;
    sidebar.classList.toggle('d-none');
}

window.addEventListener('load', () => {
    const toggleButton = document.querySelector('#sidebarToggle');
    if (toggleButton) {
        toggleButton.addEventListener('click', toggleSidebar);
    }

    const cameraSource = document.querySelector('#cameraSource');
    const cameraIndexesField = document.querySelector('#cameraIndexesField');
    const cameraUrlField = document.querySelector('#cameraUrlField');

    function updateScannerFields() {
        if (!cameraSource || !cameraIndexesField || !cameraUrlField) return;

        const selected = cameraSource.value;
        cameraIndexesField.style.display = selected === 'multiple' ? 'block' : 'none';
        cameraUrlField.style.display = selected === 'phone_url' ? 'block' : 'none';
    }

    if (cameraSource) {
        cameraSource.addEventListener('change', updateScannerFields);
        updateScannerFields();
    }
});
