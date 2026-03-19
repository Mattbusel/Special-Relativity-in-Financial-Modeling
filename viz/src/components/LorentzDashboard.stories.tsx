import type { Meta, StoryObj } from '@storybook/react';
import LorentzDashboard from './LorentzDashboard';

const meta: Meta<typeof LorentzDashboard> = {
  title: 'Components/LorentzDashboard',
  component: LorentzDashboard,
  parameters: {
    layout: 'fullscreen',
  },
  argTypes: {
    beta: {
      control: { type: 'range', min: 0, max: 0.9999, step: 0.0001 },
      description: 'Market velocity (β) — drives all four Lorentz charts and the live kinematics panel',
    },
  },
};

export default meta;
type Story = StoryObj<typeof LorentzDashboard>;

export const Default: Story = {
  args: {
    beta: 0.72,
  },
};
