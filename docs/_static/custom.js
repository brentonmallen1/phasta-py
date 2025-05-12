// Custom JavaScript for PHASTA documentation

// Add copy button to code blocks
document.addEventListener('DOMContentLoaded', function() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(function(block) {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        
        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';
        block.parentNode.insertBefore(wrapper, block);
        wrapper.appendChild(block);
        wrapper.appendChild(button);
        
        button.addEventListener('click', function() {
            const text = block.textContent;
            navigator.clipboard.writeText(text).then(function() {
                button.textContent = 'Copied!';
                setTimeout(function() {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
    });
});

// Add table of contents to long pages
document.addEventListener('DOMContentLoaded', function() {
    const content = document.querySelector('.wy-nav-content-wrap');
    if (!content) return;
    
    const headings = content.querySelectorAll('h2, h3');
    if (headings.length < 3) return;
    
    const toc = document.createElement('div');
    toc.className = 'local-toc';
    toc.innerHTML = '<h4>On This Page</h4><ul></ul>';
    
    const tocList = toc.querySelector('ul');
    headings.forEach(function(heading) {
        const item = document.createElement('li');
        const link = document.createElement('a');
        link.textContent = heading.textContent;
        link.href = '#' + heading.id;
        item.appendChild(link);
        tocList.appendChild(item);
    });
    
    content.insertBefore(toc, content.firstChild);
});

// Add search functionality
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.querySelector('.wy-side-nav-search input');
    if (!searchInput) return;
    
    searchInput.addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase();
        const content = document.querySelector('.wy-nav-content');
        const elements = content.querySelectorAll('p, li, code, pre');
        
        elements.forEach(function(element) {
            const text = element.textContent.toLowerCase();
            if (text.includes(searchTerm)) {
                element.classList.add('highlight');
            } else {
                element.classList.remove('highlight');
            }
        });
    });
});

// Add dark mode toggle
document.addEventListener('DOMContentLoaded', function() {
    const toggle = document.createElement('button');
    toggle.className = 'dark-mode-toggle';
    toggle.textContent = '🌙';
    toggle.title = 'Toggle dark mode';
    
    document.body.appendChild(toggle);
    
    toggle.addEventListener('click', function() {
        document.body.classList.toggle('dark-mode');
        toggle.textContent = document.body.classList.contains('dark-mode') ? '☀️' : '🌙';
    });
});

// Add version selector
document.addEventListener('DOMContentLoaded', function() {
    const versions = ['0.1.0', '0.2.0', 'main'];
    const currentVersion = document.querySelector('.wy-side-nav-search .version').textContent;
    
    const selector = document.createElement('select');
    selector.className = 'version-selector';
    
    versions.forEach(function(version) {
        const option = document.createElement('option');
        option.value = version;
        option.textContent = version;
        if (version === currentVersion) {
            option.selected = true;
        }
        selector.appendChild(option);
    });
    
    const search = document.querySelector('.wy-side-nav-search');
    search.appendChild(selector);
    
    selector.addEventListener('change', function(e) {
        const version = e.target.value;
        const baseUrl = window.location.href.split('/').slice(0, -2).join('/');
        window.location.href = `${baseUrl}/${version}/`;
    });
}); 