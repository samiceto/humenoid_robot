import React from 'react';
import clsx from 'clsx';
import styles from './HardwareSpec.module.css';

/**
 * A component for displaying hardware specifications in a structured format
 */
export default function HardwareSpec({title, specs}) {
  return (
    <div className={styles.hardwareSpec}>
      <div className={styles.hardwareSpecTitle}>
        {title}
      </div>
      <div className={styles.hardwareSpecDetails}>
        {specs.map((spec, index) => (
          <div key={index} className={styles.hardwareSpecItem}>
            <div className={styles.hardwareSpecLabel}>{spec.label}</div>
            <div className={styles.hardwareSpecValue}>{spec.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}