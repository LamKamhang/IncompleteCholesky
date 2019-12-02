;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((c-or-c++-mode
  (eval . (let ((root (projectile-project-root)))
            (setq-local local-project-include-path
                        (list
                         (concat root "src")
                         (concat root "external")))
            (setq-local company-c-headers-path-user
                        (append company-c-headers-path-user
                                local-project-include-path))
            (setq-local flycheck-clang-include-path
                        (append flycheck-clang-include-path
                                local-project-include-path))))))
