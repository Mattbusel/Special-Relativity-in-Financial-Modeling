import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { generateOHLCV, computeSpacetimeEvents, gamma } from '../utils/physics';
import type { SpacetimeEvent } from '../types/market';

interface LightCone3DProps {
  beta: number;
}

// ─── Cone geometry ────────────────────────────────────────────────────────────

interface ConeProps {
  beta: number;
}

function LightCones({ beta }: ConeProps) {
  const futureMeshRef = useRef<THREE.Mesh>(null);
  const pastMeshRef = useRef<THREE.Mesh>(null);
  const futureWireRef = useRef<THREE.Mesh>(null);
  const pastWireRef = useRef<THREE.Mesh>(null);
  const futureRedRef = useRef<THREE.Mesh>(null);
  const pastRedRef = useRef<THREE.Mesh>(null);

  const targetAngle = useRef(beta);
  const currentAngle = useRef(beta);

  useEffect(() => {
    targetAngle.current = beta;
  }, [beta]);

  useFrame((_, delta) => {
    // Smoothly animate cone radius toward target
    const speed = 1 / 0.5; // 0.5s transition
    const diff = targetAngle.current - currentAngle.current;
    currentAngle.current += diff * Math.min(delta * speed, 1);

    const coneRadius = 1.0 + currentAngle.current * 1.5; // opens wider as beta→1
    const coneHeight = 3;
    const segments = 32;

    const updateConeGeo = (mesh: THREE.Mesh | null, open: boolean) => {
      if (!mesh) return;
      mesh.geometry.dispose();
      const newGeo = new THREE.ConeGeometry(coneRadius, coneHeight, segments, 1, open);
      mesh.geometry = newGeo;
    };

    // Update all cones
    [futureMeshRef, futureWireRef, futureRedRef].forEach(ref => updateConeGeo(ref.current, false));
    [pastMeshRef, pastWireRef, pastRedRef].forEach(ref => updateConeGeo(ref.current, false));
  });

  const initialRadius = 1.0 + beta * 1.5;
  const coneHeight = 3;
  const segments = 32;

  return (
    <group>
      {/* Future cone — semi-transparent cyan interior */}
      <mesh ref={futureMeshRef} position={[0, coneHeight / 2, 0]}>
        <coneGeometry args={[initialRadius, coneHeight, segments]} />
        <meshPhongMaterial
          color="#00ffff"
          transparent
          opacity={0.07}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      {/* Future cone wireframe (LIGHTLIKE boundary) */}
      <mesh ref={futureWireRef} position={[0, coneHeight / 2, 0]}>
        <coneGeometry args={[initialRadius, coneHeight, segments]} />
        <meshBasicMaterial color="#ffff00" wireframe transparent opacity={0.7} />
      </mesh>

      {/* Past cone — semi-transparent cyan interior */}
      <mesh ref={pastMeshRef} position={[0, -coneHeight / 2, 0]} rotation={[Math.PI, 0, 0]}>
        <coneGeometry args={[initialRadius, coneHeight, segments]} />
        <meshPhongMaterial
          color="#00ffff"
          transparent
          opacity={0.07}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      {/* Past cone wireframe */}
      <mesh ref={pastWireRef} position={[0, -coneHeight / 2, 0]} rotation={[Math.PI, 0, 0]}>
        <coneGeometry args={[initialRadius, coneHeight, segments]} />
        <meshBasicMaterial color="#ffff00" wireframe transparent opacity={0.7} />
      </mesh>

      {/* Spacelike exterior hints — red halo rings */}
      <mesh ref={futureRedRef} position={[0, coneHeight / 2, 0]}>
        <coneGeometry args={[initialRadius * 1.6, coneHeight, segments]} />
        <meshBasicMaterial
          color="#ff3333"
          transparent
          opacity={0.03}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      <mesh ref={pastRedRef} position={[0, -coneHeight / 2, 0]} rotation={[Math.PI, 0, 0]}>
        <coneGeometry args={[initialRadius * 1.6, coneHeight, segments]} />
        <meshBasicMaterial
          color="#ff3333"
          transparent
          opacity={0.03}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

// ─── Axes ─────────────────────────────────────────────────────────────────────

function Axes() {
  const axisLen = 3.5;

  const makeAxis = (dir: [number, number, number], color: string) => {
    const points = [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(...dir).multiplyScalar(axisLen),
    ];
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    return { geo, color };
  };

  const axes = [
    makeAxis([0, 1, 0], '#ffffff'),  // Time (Y)
    makeAxis([1, 0, 0], '#00ffff'),  // Price (X)
    makeAxis([0, 0, 1], '#00ff41'), // Volume (Z)
  ];

  return (
    <group>
      {axes.map(({ geo, color }, i) => (
        <line key={i}>
          <bufferGeometry attach="geometry" {...geo} />
          <lineBasicMaterial attach="material" color={color} linewidth={2} />
        </line>
      ))}

      {/* Axis labels */}
      <Text position={[0, axisLen + 0.3, 0]} fontSize={0.18} color="#ffffff" anchorX="center">
        TIME
      </Text>
      <Text position={[axisLen + 0.3, 0, 0]} fontSize={0.18} color="#00ffff" anchorX="center">
        PRICE
      </Text>
      <Text position={[0, 0, axisLen + 0.3]} fontSize={0.18} color="#00ff41" anchorX="center">
        VOLUME
      </Text>

      {/* Origin label */}
      <Text position={[0.1, 0.1, 0.1]} fontSize={0.12} color="#888888" anchorX="left">
        O
      </Text>
    </group>
  );
}

// ─── Event dots (instanced) ────────────────────────────────────────────────────

interface DotsProps {
  events: SpacetimeEvent[];
  visibleCount: number;
}

const REGIME_COLORS: Record<string, THREE.Color> = {
  TIMELIKE: new THREE.Color('#00ffff'),
  SPACELIKE: new THREE.Color('#ff3333'),
  LIGHTLIKE: new THREE.Color('#ffff00'),
};

function EventDots({ events, visibleCount }: DotsProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  const visible = useMemo(() => events.slice(0, visibleCount), [events, visibleCount]);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    visible.forEach((evt, i) => {
      const t = i / Math.max(events.length - 1, 1);

      // Map to 3D space: time along Y, price along X, volume along Z
      const x = ((evt.bar.close - 100) / 50) * 2;
      const y = 3 - t * 6;  // descend from +3 to -3 over time
      const z = (evt.bar.volume / 5000 - 0.5) * 1.5;

      dummy.position.set(x, y, z);
      dummy.scale.setScalar(0.08);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      const color = REGIME_COLORS[evt.regime] ?? REGIME_COLORS.TIMELIKE;
      mesh.setColorAt(i, color);
    });

    mesh.count = visible.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [visible, events.length, dummy]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, events.length]}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshPhongMaterial />
    </instancedMesh>
  );
}

// ─── Auto-rotate controller ────────────────────────────────────────────────────

function AutoRotate() {
  const { camera } = useThree();
  const lastInteraction = useRef(Date.now());
  const angle = useRef(0);

  useEffect(() => {
    const onActivity = () => {
      lastInteraction.current = Date.now();
    };
    window.addEventListener('mousemove', onActivity);
    window.addEventListener('mousedown', onActivity);
    return () => {
      window.removeEventListener('mousemove', onActivity);
      window.removeEventListener('mousedown', onActivity);
    };
  }, []);

  useFrame((_, delta) => {
    const idle = (Date.now() - lastInteraction.current) / 1000;
    if (idle > 3) {
      angle.current += delta * 0.3;
      const r = 8;
      camera.position.x = Math.sin(angle.current) * r;
      camera.position.z = Math.cos(angle.current) * r;
      camera.lookAt(0, 0, 0);
    }
  });

  return null;
}

// ─── Scene ────────────────────────────────────────────────────────────────────

interface SceneProps {
  beta: number;
  events: SpacetimeEvent[];
  visibleCount: number;
}

function Scene({ beta, events, visibleCount }: SceneProps) {
  return (
    <>
      <color attach="background" args={['#0a0a0a']} />
      <ambientLight intensity={0.4} />
      <pointLight position={[5, 5, 5]} intensity={1.5} color="#00ffff" />
      <pointLight position={[-5, -5, -5]} intensity={0.8} color="#ff3333" />
      <pointLight position={[0, 8, 0]} intensity={0.6} color="#ffffff" />

      <LightCones beta={beta} />
      <Axes />
      <EventDots events={events} visibleCount={visibleCount} />

      {/* Grid floor */}
      <Grid
        position={[0, -3.2, 0]}
        args={[12, 12]}
        cellSize={0.5}
        cellThickness={0.4}
        cellColor="#1a1a2e"
        sectionSize={2}
        sectionThickness={0.8}
        sectionColor="#00ffff22"
        fadeDistance={18}
        fadeStrength={1}
        infiniteGrid
      />

      <OrbitControls
        enablePan
        enableZoom
        enableRotate
        minDistance={3}
        maxDistance={20}
        target={[0, 0, 0]}
      />
      <AutoRotate />
    </>
  );
}

// ─── Legend overlay ────────────────────────────────────────────────────────────

function Legend({ beta }: { beta: number }) {
  const g = gamma(beta);
  return (
    <div style={{
      position: 'absolute',
      top: 16,
      right: 16,
      background: 'rgba(5,5,8,0.85)',
      border: '1px solid #1a1a2e',
      borderRadius: 4,
      padding: '12px 16px',
      fontSize: '11px',
      fontFamily: 'inherit',
      lineHeight: '1.8',
      backdropFilter: 'blur(4px)',
      zIndex: 10,
    }}>
      <div style={{ color: '#888', marginBottom: 8, letterSpacing: '0.1em', fontSize: '10px' }}>
        LIGHT CONE LEGEND
      </div>
      <LegendItem color="#ffff00" label="LIGHTLIKE boundary" />
      <LegendItem color="#00ffff" label="TIMELIKE (causal)" />
      <LegendItem color="#ff3333" label="SPACELIKE (acausal)" />
      <div style={{ borderTop: '1px solid #1a1a2e', marginTop: 8, paddingTop: 8 }}>
        <div style={{ color: '#555', fontSize: '10px' }}>β = {beta.toFixed(4)}</div>
        <div style={{ color: '#555', fontSize: '10px' }}>γ = {g.toFixed(4)}</div>
        <div style={{ color: '#555', fontSize: '10px' }}>c_mkt = {(1 + beta * 0.5).toFixed(4)}</div>
      </div>
    </div>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{
        width: 10,
        height: 10,
        borderRadius: 2,
        background: color,
        boxShadow: `0 0 4px ${color}`,
        flexShrink: 0,
      }} />
      <span style={{ color: '#aaa' }}>{label}</span>
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────

export default function LightCone3D({ beta }: LightCone3DProps) {
  const [events, setEvents] = useState<SpacetimeEvent[]>([]);
  const [visibleCount, setVisibleCount] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const allEvents = useMemo(() => {
    const bars = generateOHLCV(100, 42);
    return computeSpacetimeEvents(bars, beta);
  }, [beta]);

  const startAnimation = useCallback(() => {
    setEvents(allEvents);
    setVisibleCount(0);

    if (intervalRef.current) clearInterval(intervalRef.current);

    let count = 0;
    intervalRef.current = setInterval(() => {
      count++;
      setVisibleCount(count);
      if (count >= allEvents.length) {
        if (intervalRef.current) clearInterval(intervalRef.current);
      }
    }, 100);
  }, [allEvents]);

  useEffect(() => {
    startAnimation();
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [startAnimation]);

  const timeCounts = useMemo(() => {
    const visible = allEvents.slice(0, visibleCount);
    return {
      TIMELIKE: visible.filter(e => e.regime === 'TIMELIKE').length,
      SPACELIKE: visible.filter(e => e.regime === 'SPACELIKE').length,
      LIGHTLIKE: visible.filter(e => e.regime === 'LIGHTLIKE').length,
    };
  }, [allEvents, visibleCount]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', background: '#0a0a0a' }}>
      <Canvas
        camera={{ position: [6, 3, 6], fov: 55, near: 0.1, far: 100 }}
        gl={{ antialias: true, alpha: false }}
        dpr={[1, 2]}
      >
        <Scene beta={beta} events={events} visibleCount={visibleCount} />
      </Canvas>

      <Legend beta={beta} />

      {/* Stats overlay */}
      <div style={{
        position: 'absolute',
        bottom: 16,
        left: 16,
        background: 'rgba(5,5,8,0.85)',
        border: '1px solid #1a1a2e',
        borderRadius: 4,
        padding: '10px 14px',
        fontSize: '11px',
        fontFamily: 'inherit',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        gap: 20,
      }}>
        <span style={{ color: '#00ffff' }}>TIMELIKE: {timeCounts.TIMELIKE}</span>
        <span style={{ color: '#ff3333' }}>SPACELIKE: {timeCounts.SPACELIKE}</span>
        <span style={{ color: '#ffff00' }}>LIGHTLIKE: {timeCounts.LIGHTLIKE}</span>
        <span style={{ color: '#555' }}>
          {visibleCount}/{allEvents.length} bars
        </span>
        <button
          onClick={startAnimation}
          style={{
            background: 'transparent',
            border: '1px solid #00ffff44',
            color: '#00ffff',
            fontSize: '10px',
            fontFamily: 'inherit',
            padding: '2px 8px',
            cursor: 'pointer',
            borderRadius: 2,
          }}
        >
          REPLAY
        </button>
      </div>

      {/* Instructions */}
      <div style={{
        position: 'absolute',
        top: 16,
        left: 16,
        color: '#333',
        fontSize: '10px',
        letterSpacing: '0.05em',
      }}>
        DRAG to rotate · SCROLL to zoom · AUTO-ROTATES after 3s idle
      </div>
    </div>
  );
}
