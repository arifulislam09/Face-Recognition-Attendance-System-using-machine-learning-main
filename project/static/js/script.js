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
});
