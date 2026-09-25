"""Education and learning."""

from trainjudge.domains import DomainPack

PACK = DomainPack(
    name="education",
    label="Education",
    description="Schools, universities, tutoring and ed-tech",
    terms=(
        "student",
        "students",
        "teacher",
        "teachers",
        "course",
        "courses",
        "curriculum",
        "syllabus",
        "lesson",
        "lessons",
        "homework",
        "exam",
        "exams",
        "quiz",
        "quizzes",
        "tutor",
        "tutoring",
        "learning outcomes",
        "edtech",
        "ed-tech",
        "classroom",
        "school",
        "schools",
        "university",
        "universities",
        "admissions",
        "assignment",
        "assignments",
        "rubric",
    ),
    changing_fact_terms=(
        "syllabus",
        "curriculum",
        "deadlines",
        "exam dates",
        "admission requirements",
        "tuition",
        "fees",
        "course catalog",
        "timetable",
        "schedules",
    ),
    changing_facts="new syllabi, exam dates, deadlines and admission rules",
    changing_fact_note=(
        "change every term. Serve them from the current course and admissions documents, with "
        "the term or year on each one."
    ),
    retrieval_hint=" (tag each document with its term or academic year)",
    high_stakes_terms=(
        "grading",
        "grade essays",
        "admission decision",
        "admission decisions",
        "admissions decisions",
        "plagiarism",
        "academic misconduct",
        "expulsion",
        "scholarship decision",
    ),
    high_stakes_note=(
        "Grades, admissions and misconduct findings affect students' futures. Keep an educator "
        "in the loop, let students contest outcomes, and check for bias across student groups."
    ),
    closing_notes=(
        "Student records are protected in many places (for example FERPA in the US). Remove "
        "names and student IDs before training.",
    ),
)
