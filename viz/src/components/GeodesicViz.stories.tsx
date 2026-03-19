import type { Meta, StoryObj } from '@storybook/react';
import GeodesicViz from './GeodesicViz';

const meta: Meta<typeof GeodesicViz> = {
  title: 'Components/GeodesicViz',
  component: GeodesicViz,
  parameters: {
    layout: 'fullscreen',
  },
  argTypes: {
    beta: {
      control: { type: 'range', min: 0, max: 0.9999, step: 0.0001 },
      description: 'Market velocity (β) — higher values amplify geodesic curvature and deviation signals',
    },
  },
};

export default meta;
type Story = StoryObj<typeof GeodesicViz>;

export const Default: Story = {
  args: {
    beta: 0.5,
  },
};

export const HighBeta: Story = {
  name: 'High Beta (0.9) — amplified deviation',
  args: {
    beta: 0.9,
  },
};

export const LowBeta: Story = {
  name: 'Low Beta (0.1) — minimal curvature',
  args: {
    beta: 0.1,
  },
};
