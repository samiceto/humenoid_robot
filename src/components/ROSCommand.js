import React from 'react';
import clsx from 'clsx';
import styles from './ROSCommand.module.css';

/**
 * A component for displaying ROS commands with proper formatting
 */
export default function ROSCommand({command, description, terminal = false}) {
  return (
    <div className={styles.rosCommand}>
      <div className={styles.rosCommandDescription}>
        {description && <p>{description}</p>}
      </div>
      <div className={styles.rosCommandContent}>
        <code className={styles.rosCommandCode}>
          {command}
        </code>
      </div>
      {terminal && (
        <div className={styles.terminalOutput}>
          <div className={styles.terminalPrompt}>user@robot:~$ {command}</div>
          <div className={styles.terminalOutputContent}>
            {/* Terminal output would be added dynamically or via props */}
          </div>
        </div>
      )}
    </div>
  );
}