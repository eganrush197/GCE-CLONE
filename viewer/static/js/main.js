// ABOUTME: Main application entry point for Gaussian Splat Viewer
// ABOUTME: Initializes Three.js scene, loads files, and manages UI

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ============================================================================
// Global State
// ============================================================================

const state = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    currentPoints: null,
    currentFile: null,
    files: [],
    fps: 0,
    frameCount: 0,
    lastTime: performance.now()
};

// ============================================================================
// Initialization
// ============================================================================

function init() {
    console.log('Initializing Gaussian Splat Viewer...');
    
    // Setup Three.js scene
    setupScene();
    
    // Setup UI
    setupUI();
    
    // Load file list
    loadFileList();
    
    // Start render loop
    animate();
    
    console.log('Viewer initialized successfully!');
}

function setupScene() {
    const canvas = document.getElementById('viewer-canvas');
    
    // Create scene
    state.scene = new THREE.Scene();
    state.scene.background = new THREE.Color(0x0a0a0a);
    
    // Create camera
    state.camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.01,
        1000
    );
    state.camera.position.set(2, 1, 2);
    state.camera.lookAt(0, 0, 0);
    
    // Create renderer
    state.renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true,
        alpha: false
    });
    state.renderer.setSize(window.innerWidth, window.innerHeight);
    state.renderer.setPixelRatio(window.devicePixelRatio);
    
    // Create orbit controls
    state.controls = new OrbitControls(state.camera, canvas);
    state.controls.enableDamping = true;
    state.controls.dampingFactor = 0.05;
    state.controls.screenSpacePanning = false;
    state.controls.minDistance = 0.1;
    state.controls.maxDistance = 100;
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    state.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(5, 5, 5);
    state.scene.add(directionalLight);
    
    // Add grid helper
    const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
    state.scene.add(gridHelper);
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

function setupUI() {
    // File list will be populated by loadFileList()
    // UI event handlers will be added when files are loaded

    // Setup clear console button
    document.getElementById('clear-console').addEventListener('click', clearConsole);

    // Initial console message
    logToConsole('Gaussian Splat Viewer initialized', 'info');
}

// ============================================================================
// File Loading
// ============================================================================

async function loadFileList() {
    try {
        logToConsole('Fetching file list from server...', 'info');
        const response = await fetch('/api/files');
        const data = await response.json();

        state.files = data.files;

        // Update UI
        updateFileList();

        logToConsole(`Found ${state.files.length} PLY files`, 'info');
    } catch (error) {
        logToConsole(`Failed to load file list: ${error.message}`, 'error');
        console.error('Failed to load file list:', error);
        document.getElementById('file-list').innerHTML =
            '<div class="loading" style="color: #f44336;">Failed to load files</div>';
    }
}

function updateFileList() {
    const fileListEl = document.getElementById('file-list');
    
    if (state.files.length === 0) {
        fileListEl.innerHTML = '<div class="loading">No PLY files found</div>';
        return;
    }
    
    fileListEl.innerHTML = '';
    
    state.files.forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-name">${file.name}</div>
            <div class="file-info">${formatFileSize(file.size)}</div>
        `;
        
        fileItem.addEventListener('click', () => loadFile(file.name));
        fileListEl.appendChild(fileItem);
    });
}

async function loadFile(filename) {
    logToConsole(`Loading file: ${filename}`, 'info');

    // Show loading overlay
    document.getElementById('loading-overlay').classList.remove('hidden');
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('progress-text').textContent = '0%';

    try {
        // Use binary streaming endpoint for maximum performance
        const response = await fetch(`/api/load-binary/${filename}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();

        let buffer = new Uint8Array(0);
        let totalVertices = 0;
        let loadedVertices = 0;
        let allPositions = [];
        let allSHDC = [];
        let allScales = [];
        let allRotations = [];
        let allOpacities = [];
        let metadataReceived = false;

        logToConsole(`Starting binary stream...`, 'info');

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            // Append new data to buffer
            const newBuffer = new Uint8Array(buffer.length + value.length);
            newBuffer.set(buffer);
            newBuffer.set(value, buffer.length);
            buffer = newBuffer;

            // Process buffer
            while (buffer.length > 0) {
                // First, check for metadata (JSON header)
                if (!metadataReceived && buffer.length >= 4) {
                    const metadataLength = new DataView(buffer.buffer, buffer.byteOffset, 4).getUint32(0, true);

                    if (buffer.length >= 4 + metadataLength) {
                        // Extract metadata JSON
                        const metadataBytes = buffer.slice(4, 4 + metadataLength);
                        const metadataStr = new TextDecoder().decode(metadataBytes);
                        const metadata = JSON.parse(metadataStr);

                        if (metadata.type === 'metadata') {
                            totalVertices = metadata.vertex_count;
                            logToConsole(`File size: ${formatFileSize(metadata.file_size)}`, 'info');
                            logToConsole(`Total points: ${totalVertices.toLocaleString()}`, 'info');
                            logToConsole(`Format: Binary (compressed)`, 'info');
                            metadataReceived = true;
                        }

                        // Remove metadata from buffer
                        buffer = buffer.slice(4 + metadataLength);
                        continue;
                    } else {
                        // Wait for more data
                        break;
                    }
                }

                // Process binary chunks
                if (metadataReceived && buffer.length >= 24) {
                    // Read chunk header (24 bytes)
                    const headerView = new DataView(buffer.buffer, buffer.byteOffset, 24);
                    const magic = headerView.getUint32(0, true);

                    // Check magic number
                    if (magic !== 0x47535053) {
                        // Not a valid chunk, might be error message
                        if (buffer.length >= 4) {
                            const errorLength = headerView.getUint32(0, true);
                            if (buffer.length >= 4 + errorLength) {
                                const errorBytes = buffer.slice(4, 4 + errorLength);
                                const errorStr = new TextDecoder().decode(errorBytes);
                                const errorMsg = JSON.parse(errorStr);
                                throw new Error(errorMsg.message);
                            }
                        }
                        break; // Wait for more data
                    }

                    const chunkIndex = headerView.getUint32(4, true);
                    const pointCount = headerView.getUint32(8, true);
                    const totalPoints = headerView.getUint32(12, true);
                    const progress = headerView.getFloat32(16, true);

                    const dataSize = pointCount * 56; // 14 floats per gaussian (pos + sh_dc + scale + rot + opacity)
                    const totalChunkSize = 24 + dataSize;

                    if (buffer.length >= totalChunkSize) {
                        // Extract chunk data
                        const dataBytes = buffer.slice(24, totalChunkSize);
                        const floatArray = new Float32Array(dataBytes.buffer, dataBytes.byteOffset, pointCount * 14);

                        // Extract all gaussian properties
                        // Format: [x,y,z, sh0,sh1,sh2, sx,sy,sz, rw,rx,ry,rz, opacity]
                        for (let i = 0; i < pointCount; i++) {
                            const offset = i * 14;
                            // Position (3)
                            allPositions.push(floatArray[offset], floatArray[offset + 1], floatArray[offset + 2]);
                            // SH DC (3)
                            allSHDC.push(floatArray[offset + 3], floatArray[offset + 4], floatArray[offset + 5]);
                            // Scales (3)
                            allScales.push(floatArray[offset + 6], floatArray[offset + 7], floatArray[offset + 8]);
                            // Rotation quaternion (4) - stored as w,x,y,z
                            allRotations.push(floatArray[offset + 9], floatArray[offset + 10], floatArray[offset + 11], floatArray[offset + 12]);
                            // Opacity (1)
                            allOpacities.push(floatArray[offset + 13]);
                        }

                        loadedVertices += pointCount;

                        // Update progress
                        document.getElementById('progress-fill').style.width = `${progress}%`;
                        document.getElementById('progress-text').textContent = `${progress.toFixed(1)}%`;

                        logToConsole(`Loaded chunk ${chunkIndex + 1}: ${loadedVertices.toLocaleString()} / ${totalPoints.toLocaleString()} points (${progress.toFixed(1)}%)`, 'debug');

                        // Remove processed chunk from buffer
                        buffer = buffer.slice(totalChunkSize);
                    } else {
                        // Wait for more data
                        break;
                    }
                } else {
                    // Wait for more data
                    break;
                }
            }
        }

        logToConsole(`Loading complete! Creating gaussian splats...`, 'info');

        // Diagnostic: Check gaussian data
        logToConsole(`Gaussian diagnostics:`, 'debug');
        logToConsole(`  Total gaussians: ${totalVertices.toLocaleString()}`, 'debug');
        logToConsole(`  Positions: ${allPositions.length / 3}`, 'debug');
        logToConsole(`  SH DC: ${allSHDC.length / 3}`, 'debug');
        logToConsole(`  Scales: ${allScales.length / 3}`, 'debug');
        logToConsole(`  Rotations: ${allRotations.length / 4}`, 'debug');
        logToConsole(`  Opacities: ${allOpacities.length}`, 'debug');

        // Calculate SH DC statistics
        let minSH = Infinity, maxSH = -Infinity, sumSH = 0;
        for (let i = 0; i < allSHDC.length; i++) {
            minSH = Math.min(minSH, allSHDC[i]);
            maxSH = Math.max(maxSH, allSHDC[i]);
            sumSH += allSHDC[i];
        }
        const avgSH = sumSH / allSHDC.length;

        logToConsole(`  SH DC range: [${minSH.toFixed(3)}, ${maxSH.toFixed(3)}], avg: ${avgSH.toFixed(3)}`, 'debug');
        logToConsole(`  First gaussian SH DC: [${allSHDC.slice(0, 3).map(v => v.toFixed(3)).join(', ')}]`, 'debug');

        // Create gaussian splats from accumulated data
        createGaussianSplats(allPositions, allSHDC, allScales, allRotations, allOpacities, totalVertices);

        // Update UI
        state.currentFile = filename;
        document.getElementById('current-file').textContent = filename;
        document.getElementById('point-count').textContent = totalVertices.toLocaleString();

        // Update active file in list
        document.querySelectorAll('.file-item').forEach(item => {
            item.classList.remove('active');
            if (item.querySelector('.file-name').textContent === filename) {
                item.classList.add('active');
            }
        });

        logToConsole(`Successfully loaded ${totalVertices.toLocaleString()} points!`, 'info');

    } catch (error) {
        logToConsole(`Failed to load file: ${error.message}`, 'error');
        console.error('Failed to load file:', error);
        alert(`Failed to load file: ${error.message}`);
    } finally {
        // Hide loading overlay
        document.getElementById('loading-overlay').classList.add('hidden');
    }
}

function mergeChunks(chunks, totalVertices) {
    const merged = {
        vertex_count: totalVertices,
        gaussians: {
            positions: [],
            normals: [],
            colors: [],
            opacities: [],
            scales: [],
            rotations: []
        }
    };

    for (const chunk of chunks) {
        merged.gaussians.positions.push(...chunk.positions);
        merged.gaussians.normals.push(...chunk.normals);
        merged.gaussians.colors.push(...chunk.colors);
        merged.gaussians.opacities.push(...chunk.opacities);
        merged.gaussians.scales.push(...chunk.scales);
        merged.gaussians.rotations.push(...chunk.rotations);
    }

    return merged;
}

function createGaussianPoints(data) {
    // Remove existing points
    if (state.currentPoints) {
        state.scene.remove(state.currentPoints);
        state.currentPoints.geometry.dispose();
        state.currentPoints.material.dispose();
    }

    const gaussians = data.gaussians;
    const count = data.vertex_count;

    // Create geometry
    const geometry = new THREE.BufferGeometry();

    // Convert arrays to Float32Array
    const positions = new Float32Array(gaussians.positions.flat());
    const colors = new Float32Array(gaussians.colors.flat());

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Create material
    const material = new THREE.PointsMaterial({
        size: 0.01,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.8,
        blending: THREE.NormalBlending
    });

    // Create points
    state.currentPoints = new THREE.Points(geometry, material);
    state.scene.add(state.currentPoints);

    // Center camera on the points
    centerCameraOnPoints(geometry);

    console.log('Gaussian points created successfully');
}

function createGaussianPointsFromArrays(positionsArray, colorsArray, count) {
    // DEPRECATED: Use createGaussianSplats instead
    // This function is kept for backward compatibility

    // Remove existing points
    if (state.currentPoints) {
        state.scene.remove(state.currentPoints);
        state.currentPoints.geometry.dispose();
        state.currentPoints.material.dispose();
    }

    logToConsole(`Creating point cloud with ${count.toLocaleString()} points...`, 'info');

    // Create geometry
    const geometry = new THREE.BufferGeometry();

    // Convert to Float32Array (already flat arrays)
    const positions = new Float32Array(positionsArray);
    const colors = new Float32Array(colorsArray);

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Create material
    const material = new THREE.PointsMaterial({
        size: 0.01,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.8,
        blending: THREE.NormalBlending
    });

    // Create points
    state.currentPoints = new THREE.Points(geometry, material);
    state.scene.add(state.currentPoints);

    // Center camera on the points
    centerCameraOnPoints(geometry);

    logToConsole('Point cloud created successfully!', 'info');
}

function createGaussianSplats(positionsArray, shDCArray, scalesArray, rotationsArray, opacitiesArray, count) {
    // Remove existing splats
    if (state.currentPoints) {
        state.scene.remove(state.currentPoints);
        state.currentPoints.geometry.dispose();
        state.currentPoints.material.dispose();
    }

    logToConsole(`Creating ${count.toLocaleString()} gaussian splats with custom shaders...`, 'info');

    // Create instanced geometry - one quad per gaussian
    const quadGeometry = new THREE.PlaneGeometry(1, 1);
    const instancedGeometry = new THREE.InstancedBufferGeometry();

    // Copy base quad geometry
    instancedGeometry.index = quadGeometry.index;
    instancedGeometry.attributes.position = quadGeometry.attributes.position;
    instancedGeometry.attributes.uv = quadGeometry.attributes.uv;

    // Add per-instance attributes
    instancedGeometry.setAttribute('instancePosition', new THREE.InstancedBufferAttribute(new Float32Array(positionsArray), 3));
    instancedGeometry.setAttribute('instanceSHDC', new THREE.InstancedBufferAttribute(new Float32Array(shDCArray), 3));
    instancedGeometry.setAttribute('instanceScale', new THREE.InstancedBufferAttribute(new Float32Array(scalesArray), 3));
    instancedGeometry.setAttribute('instanceRotation', new THREE.InstancedBufferAttribute(new Float32Array(rotationsArray), 4));
    instancedGeometry.setAttribute('instanceOpacity', new THREE.InstancedBufferAttribute(new Float32Array(opacitiesArray), 1));

    // Create custom shader material
    const material = createGaussianSplatMaterial();

    // Create instanced mesh
    state.currentPoints = new THREE.Mesh(instancedGeometry, material);
    state.scene.add(state.currentPoints);

    // Center camera on the splats
    const tempGeometry = new THREE.BufferGeometry();
    tempGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positionsArray), 3));
    centerCameraOnPoints(tempGeometry);
    tempGeometry.dispose();

    logToConsole('Gaussian splats created successfully!', 'info');
}

function createGaussianSplatMaterial() {
    // Vertex shader: Project 3D gaussian to 2D screen space
    const vertexShader = `
        // Per-instance attributes
        attribute vec3 instancePosition;
        attribute vec3 instanceSHDC;
        attribute vec3 instanceScale;
        attribute vec4 instanceRotation;  // Quaternion (w, x, y, z)
        attribute float instanceOpacity;

        // Varyings to fragment shader
        varying vec3 vColor;
        varying float vOpacity;
        varying vec2 vUV;
        varying vec3 vScale;

        // Quaternion to rotation matrix
        mat3 quaternionToMatrix(vec4 q) {
            // q = (w, x, y, z)
            float w = q.x, x = q.y, y = q.z, z = q.w;

            return mat3(
                1.0 - 2.0*y*y - 2.0*z*z,  2.0*x*y - 2.0*w*z,        2.0*x*z + 2.0*w*y,
                2.0*x*y + 2.0*w*z,        1.0 - 2.0*x*x - 2.0*z*z,  2.0*y*z - 2.0*w*x,
                2.0*x*z - 2.0*w*y,        2.0*y*z + 2.0*w*x,        1.0 - 2.0*x*x - 2.0*y*y
            );
        }

        void main() {
            // Convert SH DC to RGB color
            vColor = instanceSHDC + 0.5;
            vOpacity = instanceOpacity;
            vScale = instanceScale;
            vUV = uv * 2.0 - 1.0; // Convert from [0,1] to [-1,1]

            // Get rotation matrix from quaternion
            mat3 R = quaternionToMatrix(instanceRotation);

            // Create scale matrix
            mat3 S = mat3(
                instanceScale.x, 0.0, 0.0,
                0.0, instanceScale.y, 0.0,
                0.0, 0.0, instanceScale.z
            );

            // Compute 3D covariance: Σ = R * S * S^T * R^T
            mat3 RS = R * S;
            mat3 Sigma = RS * transpose(RS);

            // Transform to view space
            mat3 viewMatrix3 = mat3(viewMatrix);
            mat3 J = viewMatrix3;

            // Project to 2D: Σ' = J * Σ * J^T (take upper-left 2x2)
            mat3 T = J * Sigma * transpose(J);
            mat2 Sigma2D = mat2(T[0][0], T[0][1], T[1][0], T[1][1]);

            // Compute eigenvalues for quad size (simplified: use max scale)
            float maxScale = max(max(instanceScale.x, instanceScale.y), instanceScale.z);
            float quadSize = maxScale * 3.0; // 3 sigma coverage

            // Billboard quad facing camera
            vec3 cameraRight = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
            vec3 cameraUp = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);

            // Position quad vertex
            vec3 worldPos = instancePosition + (cameraRight * position.x + cameraUp * position.y) * quadSize;

            gl_Position = projectionMatrix * viewMatrix * vec4(worldPos, 1.0);
        }
    `;

    // Fragment shader: Evaluate gaussian and apply color
    const fragmentShader = `
        varying vec3 vColor;
        varying float vOpacity;
        varying vec2 vUV;
        varying vec3 vScale;

        void main() {
            // Gaussian falloff from center
            float dist = length(vUV);

            // Gaussian function: exp(-0.5 * dist^2)
            float gaussian = exp(-0.5 * dist * dist);

            // Apply opacity
            float alpha = gaussian * vOpacity;

            // Discard if too transparent (optimization)
            if (alpha < 0.01) discard;

            // Output color with alpha
            gl_FragColor = vec4(vColor, alpha);
        }
    `;

    // Create shader material
    const material = new THREE.ShaderMaterial({
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        transparent: true,
        depthWrite: false,
        depthTest: true,
        blending: THREE.NormalBlending,
        side: THREE.DoubleSide
    });

    return material;
}

function centerCameraOnPoints(geometry) {
    geometry.computeBoundingSphere();
    const center = geometry.boundingSphere.center;
    const radius = geometry.boundingSphere.radius;

    // Position camera to view the entire object
    const distance = radius * 2.5;
    state.camera.position.set(
        center.x + distance,
        center.y + distance * 0.5,
        center.z + distance
    );

    // Update controls target
    state.controls.target.copy(center);
    state.controls.update();
}

// ============================================================================
// Animation Loop
// ============================================================================

function animate() {
    requestAnimationFrame(animate);

    // Update controls
    state.controls.update();

    // Render scene
    state.renderer.render(state.scene, state.camera);

    // Update FPS
    updateFPS();
}

function updateFPS() {
    state.frameCount++;
    const currentTime = performance.now();
    const elapsed = currentTime - state.lastTime;

    if (elapsed >= 1000) {
        state.fps = Math.round((state.frameCount * 1000) / elapsed);
        document.getElementById('fps').textContent = state.fps;

        state.frameCount = 0;
        state.lastTime = currentTime;
    }
}

// ============================================================================
// Event Handlers
// ============================================================================

function onWindowResize() {
    state.camera.aspect = window.innerWidth / window.innerHeight;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(window.innerWidth, window.innerHeight);
}

// ============================================================================
// Console Logging
// ============================================================================

function logToConsole(message, level = 'info') {
    const consoleContent = document.getElementById('console-content');
    const timestamp = new Date().toLocaleTimeString();

    const entry = document.createElement('div');
    entry.className = 'console-entry';
    entry.innerHTML = `
        <span class="console-timestamp">[${timestamp}]</span>
        <span class="console-level-${level}">[${level.toUpperCase()}]</span>
        <span>${message}</span>
    `;

    consoleContent.appendChild(entry);
    consoleContent.scrollTop = consoleContent.scrollHeight;

    // Also log to browser console
    console.log(`[${level}] ${message}`);
}

function clearConsole() {
    document.getElementById('console-content').innerHTML = '';
}

// ============================================================================
// Utility Functions
// ============================================================================

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ============================================================================
// Start Application
// ============================================================================

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

