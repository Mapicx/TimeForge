import React, { useEffect, useRef } from "react";
import * as THREE from "three";

const EARTH_TEXTURE_URL =
  "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_atmos_2048.jpg";

export default function ThreeGlobe() {
  const containerRef = useRef();

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Clean up previous renders
    container.innerHTML = "";

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Globe - using the same setup as the first code for realistic appearance
    const geometry = new THREE.SphereGeometry(2, 64, 64);
    const textureLoader = new THREE.TextureLoader();
    const earthTexture = textureLoader.load(EARTH_TEXTURE_URL);
    const material = new THREE.MeshStandardMaterial({ map: earthTexture });
    const earth = new THREE.Mesh(geometry, material);
    scene.add(earth);

    // Realistic sun lighting - one bright side, one dark side
    const ambientLight = new THREE.AmbientLight(0x404040, 0.1); // Very dim ambient light
    scene.add(ambientLight);

    // Strong directional light to simulate the sun
    const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
    sunLight.position.set(5, 3, 5);
    scene.add(sunLight);

    // Optional: Add a subtle fill light for the dark side
    const fillLight = new THREE.DirectionalLight(0x6495ed, 0.15); // Subtle blue fill
    fillLight.position.set(-5, -2, -3);
    scene.add(fillLight);

    camera.position.z = 5;

    // Satellites with trails
    const satellites = [];
    const satelliteCount = 20;

    for (let i = 0; i < satelliteCount; i++) {
      const satGeometry = new THREE.SphereGeometry(0.03, 8, 8);
      const satMaterial = new THREE.MeshBasicMaterial({ color: 0x4ade80 });
      const satellite = new THREE.Mesh(satGeometry, satMaterial);

      const pivot = new THREE.Object3D();
      pivot.add(satellite);
      scene.add(pivot);

      // Randomize orbit
      const radius = 2.5 + Math.random() * 0.5;
      satellite.position.set(radius, 0, 0);
      pivot.rotation.x = Math.random() * Math.PI * 2;
      pivot.rotation.y = Math.random() * Math.PI * 2;

      // Create trail for each satellite
      const trailLength = 30;
      const trailGeometry = new THREE.BufferGeometry();
      const trailPositions = new Float32Array(trailLength * 3);
      const trailColors = new Float32Array(trailLength * 3);

      trailGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(trailPositions, 3)
      );
      trailGeometry.setAttribute(
        "color",
        new THREE.BufferAttribute(trailColors, 3)
      );

      const trailMaterial = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.8,
      });

      const trail = new THREE.Line(trailGeometry, trailMaterial);
      scene.add(trail);

      satellites.push({
        mesh: satellite,
        pivot: pivot,
        trail: trail,
        trailPositions: [],
        speed: Math.random() * 0.005 + 0.002,
        id: `G${(i + 1).toString().padStart(2, "0")}`,
      });
    }

    // Mouse controls - fixed to prevent jumping
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    function onMouseDown(event) {
      isDragging = true;
      previousMousePosition = { x: event.clientX, y: event.clientY };
    }

    function onMouseMove(event) {
      if (!isDragging) return;

      const deltaMove = {
        x: event.clientX - previousMousePosition.x,
        y: event.clientY - previousMousePosition.y,
      };

      earth.rotation.y += deltaMove.x * 0.005;
      earth.rotation.x += deltaMove.y * 0.005;

      previousMousePosition = { x: event.clientX, y: event.clientY };
    }

    function onMouseUp(event) {
      isDragging = false;
    }

    renderer.domElement.addEventListener("mousedown", onMouseDown, false);
    renderer.domElement.addEventListener("mousemove", onMouseMove, false);
    renderer.domElement.addEventListener("mouseup", onMouseUp, false);

    // Animation loop with trail updates
    function animate() {
      requestAnimationFrame(animate);
      earth.rotation.y += 0.0055;

      satellites.forEach((sat, index) => {
        sat.pivot.rotation.z += sat.speed;

        // Get world position of satellite
        const worldPosition = new THREE.Vector3();
        sat.mesh.getWorldPosition(worldPosition);

        // Update trail positions
        sat.trailPositions.push(worldPosition.clone());
        if (sat.trailPositions.length > 30) {
          sat.trailPositions.shift();
        }

        // Update trail geometry
        const positions = sat.trail.geometry.attributes.position.array;
        const colors = sat.trail.geometry.attributes.color.array;

        for (let i = 0; i < sat.trailPositions.length; i++) {
          const pos = sat.trailPositions[i];
          positions[i * 3] = pos.x;
          positions[i * 3 + 1] = pos.y;
          positions[i * 3 + 2] = pos.z;

          // Fade color from green to transparent
          const alpha = i / sat.trailPositions.length;
          colors[i * 3] = 0.27 * alpha; // Green R
          colors[i * 3 + 1] = 0.77 * alpha; // Green G
          colors[i * 3 + 2] = 0.31 * alpha; // Green B
        }

        // Clear unused positions
        for (let i = sat.trailPositions.length; i < 30; i++) {
          positions[i * 3] = 0;
          positions[i * 3 + 1] = 0;
          positions[i * 3 + 2] = 0;
          colors[i * 3] = 0;
          colors[i * 3 + 1] = 0;
          colors[i * 3 + 2] = 0;
        }

        sat.trail.geometry.attributes.position.needsUpdate = true;
        sat.trail.geometry.attributes.color.needsUpdate = true;
        sat.trail.geometry.setDrawRange(0, sat.trailPositions.length);
      });

      renderer.render(scene, camera);
    }
    animate();

    // Handle window resize
    const handleResize = () => {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };

    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize);
      renderer.domElement.removeEventListener("mousedown", onMouseDown, false);
      renderer.domElement.removeEventListener("mousemove", onMouseMove, false);
      renderer.domElement.removeEventListener("mouseup", onMouseUp, false);
      renderer.dispose();
      container.innerHTML = "";
    };
  }, []);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <div
        ref={containerRef}
        id="globe-container"
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
