import React from 'react';
import clsx from 'clsx';
import styles from './SimulationEnv.module.css';

/**
 * A component for displaying simulation environment information
 */
export default function SimulationEnv({title, description, image, children}) {
  return (
    <div className={styles.simulationEnv}>
      <div className={styles.simulationEnvTitle}>
        {title}
      </div>
      {description && <p className={styles.simulationEnvDescription}>{description}</p>}
      {image && <img src={image} alt={title} className={styles.simulationEnvImage} />}
      {children && <div className={styles.simulationEnvContent}>{children}</div>}
    </div>
  );
}