import React from 'react';
import clsx from 'clsx';
import styles from './RoboticsBlock.module.css';

/**
 * A custom component for displaying robotics-specific content blocks
 * with a distinctive styling and icon
 */
export default function RoboticsBlock({title, children, type = 'info'}) {
  const blockType = type.toLowerCase();

  return (
    <div className={clsx(styles.roboticsBlock, styles[blockType])}>
      <div className={styles.roboticsBlockTitle}>
        {blockType === 'warning' && '⚠️ '}
        {blockType === 'note' && '📝 '}
        {blockType === 'tip' && '💡 '}
        {blockType === 'important' && '❗ '}
        {title}
      </div>
      <div className={styles.roboticsBlockContent}>
        {children}
      </div>
    </div>
  );
}