const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3000;
const PUBLIC_DIR = path.join(__dirname, 'experiment');

const MIME_TYPES = {
    '.html': 'text/html',
    '.css': 'text/css',
    '.js': 'text/javascript',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon'
};

const server = http.createServer((req, res) => {
    console.log(`${req.method} ${req.url}`);
    
    // Strip query parameters (like ?v=2) from the URL
    const urlPath = req.url.split('?')[0];
    
    // Normalize URL path to prevent directory traversal
    let filePath = path.join(PUBLIC_DIR, urlPath === '/' ? 'index.html' : urlPath);
    
    // Check if path is inside public directory
    if (!filePath.startsWith(PUBLIC_DIR)) {
        res.statusCode = 403;
        res.end('Forbidden');
        return;
    }
    
    const ext = path.extname(filePath);
    let contentType = MIME_TYPES[ext] || 'application/octet-stream';
    
    fs.readFile(filePath, (err, content) => {
        if (err) {
            if (err.code === 'ENOENT') {
                res.statusCode = 404;
                res.end('File Not Found');
            } else {
                res.statusCode = 500;
                res.end(`Internal Server Error: ${err.code}`);
            }
        } else {
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
        }
    });
});

server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}/`);
    console.log(`Serving files from: ${PUBLIC_DIR}`);
});
