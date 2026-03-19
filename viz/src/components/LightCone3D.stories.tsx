import type { Meta, StoryObj } from '@storybook/react';
import LightCone3D from './LightCone3D';

const meta: Meta<typeof LightCone3D> = {
  title: 'Components/LightCone3D',
  component: LightCone3D,
  // LightCone3D uses a Three.js Canvas that requires an explicit height on its
  // container — fullscreen layout provides this automatically.
  parameters: {
    layout: 'fullscreen',
  },
  decorators: [
    (Story) => (
      <div style={{ width: '100vw', height: '100vh', background: '#0a0a0a' }}>
        <Story />
      </div>
    ),
  ],
  argTypes: {
    beta: {
      control: { type: 'range', min: 0, max: 0.9999, step: 0.0001 },
      description: 'Market velocity relative to speed of light (β) — widens the light cone',
    },
  },
};

export default meta;
type Story = StoryObj<typeof LightCone3D>;

export const Default: Story = {
  args: {
    beta: 0.5,
  },
};

export const HighBeta: Story = {
  name: 'High Beta (0.9) — wide cone',
  args: {
    beta: 0.9,
  },
};

export const LowBeta: Story = {
  name: 'Low Beta (0.1) — narrow cone',
  args: {
    beta: 0.1,
  },
};
