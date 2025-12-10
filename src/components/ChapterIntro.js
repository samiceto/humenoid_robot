import React from 'react';
import clsx from 'clsx';
import styles from './ChapterIntro.module.css';

/**
 * A component for chapter introductions with learning objectives
 */
export default function ChapterIntro({title, subtitle, objectives}) {
  return (
    <div className={styles.chapterIntro}>
      <h1 className={styles.chapterIntroTitle}>{title}</h1>
      {subtitle && <div className={styles.chapterIntroSubtitle}>{subtitle}</div>}

      {objectives && objectives.length > 0 && (
        <div className={styles.chapterObjectives}>
          <h3 className={styles.chapterObjectivesTitle}>Learning Objectives</h3>
          <ul className={styles.chapterObjectivesList}>
            {objectives.map((objective, index) => (
              <li key={index} className={styles.chapterObjectiveItem}>
                {objective}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}