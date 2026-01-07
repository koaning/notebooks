import * as Matter from "https://esm.sh/matter-js@0.19.0";

export function render({ model, el }) {
  const container = document.createElement("div");
  container.className = "humanoid-walker-widget";
  el.appendChild(container);

  const canvas = document.createElement("canvas");
  canvas.className = "humanoid-walker-canvas";
  container.appendChild(canvas);

  // Module aliases
  const Engine = Matter.Engine,
    Render = Matter.Render,
    Runner = Matter.Runner,
    Bodies = Matter.Bodies,
    Composite = Matter.Composite,
    Constraint = Matter.Constraint,
    Body = Matter.Body,
    Vector = Matter.Vector;

  // Create engine
  const engine = Engine.create();
  const world = engine.world;

  // Create renderer
  const render = Render.create({
    element: container, // Use container, but we might need to handle canvas manually if we want custom drawing
    canvas: canvas,
    engine: engine,
    options: {
      width: 800,
      height: 400,
      wireframes: false,
      background: '#f0f0f0'
    }
  });

  // --- Create Humanoid ---
  // Dimensions
  const torsoWidth = 40, torsoHeight = 80;
  const headRadius = 15;
  const limbWidth = 15, limbLength = 60;
  
  const startX = 200, startY = 200;
  
  // Collision filters to prevent self-collision overlap where appropriate
  // Category 1: Default
  // Category 2: Body parts that shouldn't collide with each other?
  // Matter.js defaults are usually fine if we position them right, but let's use groups.
  const group = Body.nextGroup(true);

  // Bodies
  const torso = Bodies.rectangle(startX, startY, torsoWidth, torsoHeight, { 
    collisionFilter: { group: group },
    label: 'torso',
    render: { fillStyle: '#3498db' }
  });
  
  const head = Bodies.circle(startX, startY - torsoHeight/2 - headRadius, headRadius, {
    collisionFilter: { group: group },
    label: 'head',
    render: { fillStyle: '#e74c3c' }
  });

  // Legs
  const createLeg = (x, y, side) => {
    const color = side === 'left' ? '#2ecc71' : '#27ae60';
    
    const thigh = Bodies.rectangle(x, y + limbLength/2, limbWidth, limbLength, {
      collisionFilter: { group: group },
      label: `${side}_thigh`,
      render: { fillStyle: color }
    });
    
    const calf = Bodies.rectangle(x, y + limbLength * 1.5 + 5, limbWidth, limbLength, {
      collisionFilter: { group: group },
      label: `${side}_calf`,
      friction: 1.0, // High friction for feet
      render: { fillStyle: color }
    });

    return { thigh, calf };
  };

  const leftLeg = createLeg(startX - 10, startY + torsoHeight/2, 'left');
  const rightLeg = createLeg(startX + 10, startY + torsoHeight/2, 'right');

  // Constraints (Joints)
  const neck = Constraint.create({
    bodyA: torso,
    bodyB: head,
    pointA: { x: 0, y: -torsoHeight/2 },
    pointB: { x: 0, y: headRadius },
    stiffness: 0.9,
    length: 0
  });

  const createLegJoints = (leg, side, xOffset) => {
    // Hip
    const hip = Constraint.create({
      bodyA: torso,
      bodyB: leg.thigh,
      pointA: { x: xOffset, y: torsoHeight/2 },
      pointB: { x: 0, y: -limbLength/2 },
      stiffness: 0.9,
      length: 0
    });

    // Knee
    const knee = Constraint.create({
      bodyA: leg.thigh,
      bodyB: leg.calf,
      pointA: { x: 0, y: limbLength/2 },
      pointB: { x: 0, y: -limbLength/2 },
      stiffness: 0.9,
      length: 0
    });

    return { hip, knee };
  };

  const leftJoints = createLegJoints(leftLeg, 'left', -10);
  const rightJoints = createLegJoints(rightLeg, 'right', 10);

  // Floor
  const floor = Bodies.rectangle(400, 390, 810, 60, { 
    isStatic: true,
    render: { fillStyle: '#7f8c8d' },
    label: 'floor'
  });

  Composite.add(world, [
    torso, head, 
    leftLeg.thigh, leftLeg.calf, 
    rightLeg.thigh, rightLeg.calf,
    neck, 
    leftJoints.hip, leftJoints.knee,
    rightJoints.hip, rightJoints.knee,
    floor
  ]);

  // Add mouse control
  const Mouse = Matter.Mouse,
      MouseConstraint = Matter.MouseConstraint;

  const mouse = Mouse.create(render.canvas);
  const mouseConstraint = MouseConstraint.create(engine, {
      mouse: mouse,
      constraint: {
          stiffness: 0.2,
          render: {
              visible: false
          }
      }
  });
  Composite.add(world, mouseConstraint);
  render.mouse = mouse;

  // Run
  Render.run(render);
  const runner = Runner.create();
  Runner.run(runner, engine);

  // --- Logic ---

  // Handle Reset
  model.on("change:reset", () => {
    if (model.get("reset")) {
      // Reset positions
      Body.setPosition(torso, { x: startX, y: startY });
      Body.setVelocity(torso, { x: 0, y: 0 });
      Body.setAngularVelocity(torso, 0);
      Body.setAngle(torso, 0);

      Body.setPosition(head, { x: startX, y: startY - torsoHeight/2 - headRadius });
      Body.setVelocity(head, { x: 0, y: 0 });
      Body.setAngle(head, 0);

      // Helper to reset leg
      const resetLeg = (leg, xOffset) => {
         Body.setPosition(leg.thigh, { x: startX + xOffset, y: startY + torsoHeight/2 + limbLength/2 });
         Body.setVelocity(leg.thigh, { x: 0, y: 0 });
         Body.setAngle(leg.thigh, 0);
         Body.setAngularVelocity(leg.thigh, 0);

         Body.setPosition(leg.calf, { x: startX + xOffset, y: startY + torsoHeight/2 + limbLength * 1.5 + 5 });
         Body.setVelocity(leg.calf, { x: 0, y: 0 });
         Body.setAngle(leg.calf, 0);
         Body.setAngularVelocity(leg.calf, 0);
      };

      resetLeg(leftLeg, -10);
      resetLeg(rightLeg, 10);

      model.set("reset", false);
      model.save_changes();
    }
  });

  // Apply Torques
  // Listen to engine update events
  Matter.Events.on(engine, 'beforeUpdate', function(event) {
    const torques = model.get("torques"); // { left_hip, left_knee, right_hip, right_knee }
    if (!torques) return;

    // Helper to apply torque between two bodies
    // Positive torque: BodyA rotates clockwise, BodyB rotates counter-clockwise (relative)
    // Actually, simple torque on the limb body relative to parent is easier.
    // Ideally we apply equal and opposite torques.

    const applyJointTorque = (bodyA, bodyB, torque) => {
        // Torque is simple number
        if (!torque) return;
        
        // Limit max torque to avoid explosion
        const maxTorque = 1000; // Arbitrary limit
        torque = Math.max(-maxTorque, Math.min(maxTorque, torque));

        // Matter.js doesn't have a direct "apply torque relative" but applying torque to bodies works
        // A positive torque on the joint should flex/extend it.
        // Let's say positive extends.
        
        Body.setAngularVelocity(bodyA, bodyA.angularVelocity - torque * 0.0001); // Reaction
        Body.setAngularVelocity(bodyB, bodyB.angularVelocity + torque * 0.0001); // Action
        
        // Using setAngularVelocity is not physics-accurate for force, use applyForce or torque property?
        // Matter.Body.setAngularVelocity overrides physics.
        // Try direct torque modification if possible?
        // Body.torque is accumulated.
        
        // bodyA.torque -= torque;
        // bodyB.torque += torque;
        // Note: Matter.js resets force/torque every step.
        // So we need to set it every step.
    };
    
    // Better way: Apply force at a point. 
    // But let's try modifying angular velocity or just `body.torque`
    // In Matter.js, `body.torque` can be set.
    
    const scale = 5.0; // Tuning factor

    if (torques.left_hip) {
        torso.torque -= torques.left_hip * scale;
        leftLeg.thigh.torque += torques.left_hip * scale;
    }
    if (torques.left_knee) {
        leftLeg.thigh.torque -= torques.left_knee * scale;
        leftLeg.calf.torque += torques.left_knee * scale;
    }
    if (torques.right_hip) {
        torso.torque -= torques.right_hip * scale;
        rightLeg.thigh.torque += torques.right_hip * scale;
    }
    if (torques.right_knee) {
        rightLeg.thigh.torque -= torques.right_knee * scale;
        rightLeg.calf.torque += torques.right_knee * scale;
    }
  });

  // Send State back to Python
  let frameCount = 0;
  Matter.Events.on(engine, 'afterUpdate', function() {
    frameCount++;
    if (frameCount % 5 === 0) { // Every 5 frames (~12 updates/sec if 60fps)
        const state = {
            torso_x: torso.position.x,
            torso_y: torso.position.y,
            torso_angle: torso.angle,
            left_thigh_angle: leftLeg.thigh.angle,
            left_calf_angle: leftLeg.calf.angle,
            right_thigh_angle: rightLeg.thigh.angle,
            right_calf_angle: rightLeg.calf.angle
            // Can add velocities if needed
        };
        model.set("state", state);
        model.save_changes();
    }
  });
  
  // Clean up
  return () => {
    Render.stop(render);
    Runner.stop(runner);
    if (container.parentNode) container.parentNode.removeChild(container);
  };
}
